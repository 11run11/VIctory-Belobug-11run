//! Original code written for the XboxCompatibility project (task T-0038);
//! not adapted from any other repository or title.
//! SPDX-License-Identifier: GPL-3.0-or-later
//!
//! A synthetic foliage scene on the software adapter, rendered through the
//! runtime's indirect-argument clamp (`super::super`): alpha-tested grass
//! cards, instanced per patch, and tree billboards, under a fixed camera
//! that pans slowly. Every draw is an indirect draw whose record the clamp
//! bounds and charges to the list's budget, as `crate::draw` does for a
//! title's draws. Each draw writes its own number into an R32_UINT target,
//! so consecutive frames can be compared draw by draw:
//!
//! * whole-draw disappearance: a draw the scene asks for (on screen, with
//!   instances) that covers no pixel;
//! * aligned differences: pixels that change between consecutive frames
//!   once the integer pan is undone, among pixels whose draw covers pixels
//!   in both frames. They are not classified further: this host-built,
//!   single-sample pipeline does not exercise the runtime's alpha-to-
//!   coverage, mip or MSAA translation, so edge shimmer is not tested here.
//!
//! Earlier passes of the same list (a depth prepass and shadow cascades of
//! the full-density field, which the harness charges through the clamp but
//! does not rasterize: their targets are not inspected) charge most of the
//! budget first. The scene is sized so that its list totals just over 2^27
//! vertex instances: it demonstrates the budget's visible effect, not how
//! often a title's lists reach it.
//!
//! In the Belobug tree (local-test/belobug-4eb6cfd) the clamp also has a
//! rest rule (`VICTORY_INDIRECT_REST`): with it (the default) a record that
//! does not fit takes what is left of the budget, without it the record
//! alone runs empty. The scene is measured under both rules.
//! Run by `workbench/victory-runtime/belobug_foliage_test.py`.

use super::*;
use crate::host::transition;
use windows::Win32::Graphics::Dxgi::Common::{DXGI_FORMAT_R32_UINT, DXGI_FORMAT_UNKNOWN};

const W: u32 = 192;
const H: u32 = 96;
const PATCH: f32 = 16.0;
const PATCHES: usize = 32;
const TREES: usize = 8;
const RECORDS: usize = PATCHES + TREES;
/// Grass blades of a patch at the centre of the view.
const NEAR_BLADES: f32 = 6000.0;
/// Pixels the camera moves each frame.
const PAN_STEP: u32 = 3;
const FRAMES: usize = 16;
/// Shadow-cascade records charged ahead of the main pass.
const CASCADES: usize = 4;

const SCENE: &str = r#"
#define RS "RootConstants(num32BitConstants=4, b0)"
cbuffer Draw : register(b0) { uint id; uint tree; float pan; float origin; };
struct V { float4 at : SV_Position; float2 uv : UV; };
[RootSignature(RS)]
V vs(uint v : SV_VertexID, uint i : SV_InstanceID) {
    // Two triangles: (0,0) (1,0) (0,1) and (0,1) (1,0) (1,1).
    float2 c = float2((v == 1 || v == 4 || v == 5) ? 1 : 0, (v == 2 || v == 3 || v == 5) ? 1 : 0);
    float2 size;
    float2 base;
    if (tree != 0) {
        size = float2(28, 44);
        base = float2(origin, 70);
    } else {
        // Blade placement: a golden-ratio step across the patch, and rows
        // and heights from small residues of the blade number.
        float across = frac(float(i) * 0.6180340 + float(id) * 0.1370);
        size = float2(3, 7 + float((i * 7u + id) % 16u));
        base = float2(origin + across * 16.0, 72 + float((i * 5u + id * 3u) % 16u));
    }
    float2 px = float2(base.x - pan + (c.x - 0.5) * size.x, base.y - c.y * size.y);
    V o;
    o.uv = c;
    o.at = float4(px.x / 192.0 * 2 - 1, 1 - px.y / 96.0 * 2, 0.5, 1);
    return o;
}
[RootSignature(RS)]
uint ps(V o) : SV_Target {
    float a;
    if (tree != 0) {
        float2 d = o.uv - float2(0.5, 0.62);
        a = (dot(d, d) < 0.14 || (abs(o.uv.x - 0.5) < 0.08 && o.uv.y < 0.3)) ? 1 : 0;
    } else {
        // A tapered blade: alpha falls off across the card and up it.
        a = 1 - abs(o.uv.x - 0.5) * 2 - o.uv.y * 0.8;
    }
    clip(a - 0.5);
    return id + 1;
}
"#;

fn compile(source: &str, entry: windows::core::PCSTR, target: windows::core::PCSTR) -> Vec<u8> {
    let mut code: Option<ID3DBlob> = None;
    let mut errors: Option<ID3DBlob> = None;
    // SAFETY: the source and names outlive the call; out pointers are ours.
    unsafe {
        let made = D3DCompile(source.as_ptr().cast(), source.len(), s!("foliage"), None, None, entry, target, 0, 0, &mut code, Some(&mut errors));
        if made.is_err() {
            let text = errors.map(|e| {
                String::from_utf8_lossy(std::slice::from_raw_parts(e.GetBufferPointer().cast::<u8>(), e.GetBufferSize())).into_owned()
            });
            panic!("foliage shader: {text:?}");
        }
        let code = code.unwrap();
        std::slice::from_raw_parts(code.GetBufferPointer().cast::<u8>(), code.GetBufferSize()).to_vec()
    }
}

/// What the scene asks for in one frame: the main pass's records (vertex
/// count, instances) in draw order, their origins, and the earlier passes'
/// records.
struct Frame {
    pan: u32,
    main: Vec<(u32, u32)>,
    origins: Vec<f32>,
    earlier: Vec<(u32, u32)>,
}

fn patch_origin(k: usize) -> f32 {
    k as f32 * PATCH
}

fn tree_origin(t: usize) -> f32 {
    40.0 + 60.0 * t as f32
}

/// The scene's own culling and distance LOD, standing in for a title's
/// culling pass: instances of a patch or tree outside the view are zero.
fn frame(f: usize, earlier_base: u32) -> Frame {
    let pan = 8 + PAN_STEP * f as u32;
    let (left, right) = (pan as f32, (pan + W) as f32);
    let centre = (left + right) / 2.0;
    let mut main = Vec::with_capacity(RECORDS);
    let mut origins = Vec::with_capacity(RECORDS);
    let mut shadow_weight = 0.0f32;
    for k in 0..PATCHES {
        let o = patch_origin(k);
        // At least four columns of the patch on screen.
        let visible = o + PATCH - 4.0 > left && o + 4.0 < right;
        let d = (o + PATCH / 2.0 - centre).abs();
        let weight = (1.15 - d / (W as f32 / 2.0)).clamp(0.25, 1.0);
        let blades = if visible { (NEAR_BLADES * weight).round() as u32 } else { 0 };
        // The near cascade holds patches within two view widths.
        if d < W as f32 * 2.0 {
            shadow_weight += weight;
        }
        main.push((6, blades));
        origins.push(o);
    }
    for t in 0..TREES {
        let o = tree_origin(t);
        // The canopy (10.5 pixels either side) at least two columns on screen.
        let visible = o + 8.0 > left && o - 8.0 < right;
        main.push((6, u32::from(visible)));
        origins.push(o);
    }
    // Full-density shadow casters: a fixed far field per cascade and the
    // near cascade's patches at 40 blades for each blade drawn.
    let near = (shadow_weight * NEAR_BLADES * 40.0).round() as u32;
    let mut earlier = vec![(6u32, earlier_base / 6 / CASCADES as u32); CASCADES];
    earlier[0].1 += near / CASCADES as u32;
    Frame { pan, main, origins, earlier }
}

fn cost(records: &[(u32, u32)]) -> u64 {
    records.iter().map(|&(v, i)| u64::from(v) * u64::from(i)).sum()
}

/// The far field's size: frame 0's list then needs half its main pass more
/// than 2^27 vertex instances.
fn earlier_base() -> u32 {
    let probe = frame(0, 0);
    let target = (1u64 << 27) - cost(&probe.earlier) - cost(&probe.main) / 2;
    target as u32
}

struct Scene {
    root: ID3D12RootSignature,
    pso: ID3D12PipelineState,
    target: ID3D12Resource,
    rtv: ID3D12DescriptorHeap,
    signature: ID3D12CommandSignature,
}

impl Scene {
    fn new(warp: &WarpHarness) -> Self {
        let (vs, ps) = (compile(SCENE, s!("vs"), s!("vs_5_1")), compile(SCENE, s!("ps"), s!("ps_5_1")));
        let code = |bytes: &[u8]| D3D12_SHADER_BYTECODE { pShaderBytecode: bytes.as_ptr().cast(), BytecodeLength: bytes.len() };
        // SAFETY: plain creation; descriptions point at live data.
        unsafe {
            let root: ID3D12RootSignature = warp.device.CreateRootSignature(0, &vs).unwrap();
            let mut blend = D3D12_BLEND_DESC::default();
            blend.RenderTarget[0].RenderTargetWriteMask = 0xf;
            let mut formats = [DXGI_FORMAT_UNKNOWN; 8];
            formats[0] = DXGI_FORMAT_R32_UINT;
            let desc = D3D12_GRAPHICS_PIPELINE_STATE_DESC {
                pRootSignature: std::mem::transmute_copy(&root),
                VS: code(&vs),
                PS: code(&ps),
                BlendState: blend,
                SampleMask: u32::MAX,
                RasterizerState: D3D12_RASTERIZER_DESC {
                    FillMode: D3D12_FILL_MODE_SOLID,
                    CullMode: D3D12_CULL_MODE_NONE,
                    DepthClipEnable: true.into(),
                    ..Default::default()
                },
                PrimitiveTopologyType: D3D12_PRIMITIVE_TOPOLOGY_TYPE_TRIANGLE,
                NumRenderTargets: 1,
                RTVFormats: formats,
                SampleDesc: DXGI_SAMPLE_DESC { Count: 1, Quality: 0 },
                ..Default::default()
            };
            let pso = warp.device.CreateGraphicsPipelineState(&desc).unwrap();
            let mut target: Option<ID3D12Resource> = None;
            warp.device
                .CreateCommittedResource(
                    &D3D12_HEAP_PROPERTIES { Type: D3D12_HEAP_TYPE_DEFAULT, ..Default::default() },
                    D3D12_HEAP_FLAG_NONE,
                    &D3D12_RESOURCE_DESC {
                        Dimension: D3D12_RESOURCE_DIMENSION_TEXTURE2D,
                        Width: u64::from(W),
                        Height: H,
                        DepthOrArraySize: 1,
                        MipLevels: 1,
                        Format: DXGI_FORMAT_R32_UINT,
                        SampleDesc: DXGI_SAMPLE_DESC { Count: 1, Quality: 0 },
                        Flags: D3D12_RESOURCE_FLAG_ALLOW_RENDER_TARGET,
                        ..Default::default()
                    },
                    D3D12_RESOURCE_STATE_RENDER_TARGET,
                    None,
                    &mut target,
                )
                .unwrap();
            let target = target.unwrap();
            let rtv: ID3D12DescriptorHeap = warp
                .device
                .CreateDescriptorHeap(&D3D12_DESCRIPTOR_HEAP_DESC { Type: D3D12_DESCRIPTOR_HEAP_TYPE_RTV, NumDescriptors: 1, ..Default::default() })
                .unwrap();
            warp.device.CreateRenderTargetView(&target, None, rtv.GetCPUDescriptorHandleForHeapStart());
            let signature = warp.signature(D3D12_INDIRECT_ARGUMENT_TYPE_DRAW.0 as u32, 16);
            Self { root, pso, target, rtv, signature }
        }
    }
}

/// Renders one frame through `clamp`: the earlier passes' records and then
/// the main pass's are clamped into one budget, and each main record is
/// drawn by an indirect draw of its clamped copy. The picture: a draw
/// number + 1 (0: nothing) per pixel.
fn render(warp: &mut WarpHarness, scene: &Scene, clamp: &Clamp, f: &Frame) -> Vec<u32> {
    let write = |records: &[(u32, u32)]| {
        let (buffer, data, _) = warp.shared_buffer((records.len() * 16) as u64);
        // SAFETY: the mapping holds 16 bytes a record.
        unsafe {
            for (r, &(v, i)) in records.iter().enumerate() {
                for (w, word) in [v, i, 0, 0].into_iter().enumerate() {
                    data.cast::<u32>().add(r * 4 + w).write_volatile(word);
                }
            }
        }
        buffer
    };
    let (earlier_src, main_src) = (write(&f.earlier), write(&f.main));
    let (_budget, budget_ptr, budget_gpu) = warp.shared_buffer(256);
    let pitch = W * 4;
    let (back, back_ptr, _) = warp.shared_buffer(u64::from(pitch * H));
    let (earlier_dst, main_dst) = (warp.default_uav(f.earlier.len() as u64 * 16), warp.default_uav(RECORDS as u64 * 16));
    // SAFETY: the list has not read the budget yet; it starts empty.
    unsafe { std::ptr::write_bytes(budget_ptr, 0, 256) };
    for (src, dst, n) in [(&earlier_src, &earlier_dst, f.earlier.len()), (&main_src, &main_dst, RECORDS)] {
        assert!(record_clamp_as(clamp, &warp.list, (dst, D3D12_RESOURCE_STATE_COMMON), budget_gpu, (src, 0), None, (0, 16, n as u32), false));
    }
    // SAFETY: an open list; live resources; the clamped records are
    // readable as arguments (`record_clamp_as`).
    unsafe {
        let l = &warp.list;
        let rtv = scene.rtv.GetCPUDescriptorHandleForHeapStart();
        l.ClearRenderTargetView(rtv, &[0.0f32; 4], None);
        l.OMSetRenderTargets(1, Some(&rtv as *const _), false, None);
        l.RSSetViewports(&[D3D12_VIEWPORT { TopLeftX: 0.0, TopLeftY: 0.0, Width: W as f32, Height: H as f32, MinDepth: 0.0, MaxDepth: 1.0 }]);
        l.RSSetScissorRects(&[windows::Win32::Foundation::RECT { left: 0, top: 0, right: W as i32, bottom: H as i32 }]);
        l.SetGraphicsRootSignature(&scene.root);
        l.SetPipelineState(&scene.pso);
        l.IASetPrimitiveTopology(windows::Win32::Graphics::Direct3D::D3D_PRIMITIVE_TOPOLOGY_TRIANGLELIST);
        for r in 0..RECORDS {
            let constants = [r as u32, u32::from(r >= PATCHES), (f.pan as f32).to_bits(), f.origins[r].to_bits()];
            l.SetGraphicsRoot32BitConstants(0, 4, constants.as_ptr().cast(), 0);
            l.ExecuteIndirect(&scene.signature, 1, &main_dst, r as u64 * 16, None, 0);
        }
        l.ResourceBarrier(&[transition(&scene.target, D3D12_RESOURCE_STATE_RENDER_TARGET, D3D12_RESOURCE_STATE_COPY_SOURCE)]);
        let to = D3D12_TEXTURE_COPY_LOCATION {
            pResource: std::mem::transmute_copy(&back),
            Type: D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT,
            Anonymous: D3D12_TEXTURE_COPY_LOCATION_0 {
                PlacedFootprint: D3D12_PLACED_SUBRESOURCE_FOOTPRINT {
                    Offset: 0,
                    Footprint: D3D12_SUBRESOURCE_FOOTPRINT { Format: DXGI_FORMAT_R32_UINT, Width: W, Height: H, Depth: 1, RowPitch: pitch },
                },
            },
        };
        let from = D3D12_TEXTURE_COPY_LOCATION {
            pResource: std::mem::transmute_copy(&scene.target),
            Type: D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX,
            Anonymous: D3D12_TEXTURE_COPY_LOCATION_0 { SubresourceIndex: 0 },
        };
        l.CopyTextureRegion(&to, 0, 0, 0, &from, None);
        l.ResourceBarrier(&[
            transition(&scene.target, D3D12_RESOURCE_STATE_COPY_SOURCE, D3D12_RESOURCE_STATE_RENDER_TARGET),
            transition(&earlier_dst, D3D12_RESOURCE_STATE_INDIRECT_ARGUMENT, D3D12_RESOURCE_STATE_COMMON),
            transition(&main_dst, D3D12_RESOURCE_STATE_INDIRECT_ARGUMENT, D3D12_RESOURCE_STATE_COMMON),
        ]);
    }
    warp.submit_and_wait();
    // SAFETY: the list has run; the mapping holds the picture.
    unsafe { std::slice::from_raw_parts(back_ptr.cast::<u32>().cast_const(), (W * H) as usize).to_vec() }
}

/// Pixels each draw covers in a picture.
fn coverage(picture: &[u32]) -> [u32; RECORDS] {
    let mut covered = [0u32; RECORDS];
    for &p in picture {
        if p > 0 {
            covered[(p - 1) as usize] += 1;
        }
    }
    covered
}

/// The frame comparison's findings over a run of frames.
#[derive(Debug, Default)]
struct Findings {
    /// (frame, draws asked for that cover no pixel).
    vanished: Vec<(usize, Vec<usize>)>,
    /// Per consecutive pair: (pixels that differ, pixels compared), pan
    /// undone, counting only background and draws that cover pixels in both
    /// frames. Not classified further (see the module notes).
    aligned: Vec<(u32, u32)>,
}

fn compare(frames: &[Frame], pictures: &[Vec<u32>]) -> Findings {
    let mut found = Findings::default();
    for (n, (f, picture)) in frames.iter().zip(pictures).enumerate() {
        let covered = coverage(picture);
        // A draw the scene asks for is one with instances whose footprint
        // is on screen (`frame`'s culling).
        let gone: Vec<usize> = (0..RECORDS).filter(|&r| f.main[r].1 > 0 && covered[r] == 0).collect();
        if !gone.is_empty() {
            found.vanished.push((n, gone));
        }
        if n > 0 {
            let earlier = coverage(&pictures[n - 1]);
            let common = |id: u32| id == 0 || (earlier[(id - 1) as usize] > 0 && covered[(id - 1) as usize] > 0);
            let shift = (f.pan - frames[n - 1].pan) as usize;
            let before = &pictures[n - 1];
            let (mut changed, mut compared) = (0, 0);
            for y in 0..H as usize {
                for x in 0..W as usize - shift {
                    let (a, b) = (before[y * W as usize + x + shift], picture[y * W as usize + x]);
                    if common(a) && common(b) {
                        compared += 1;
                        changed += u32::from(a != b);
                    }
                }
            }
            found.aligned.push((changed, compared));
        }
    }
    found
}

/// Whole-draw flicker: (frame, draw, its pixels in the frame before, its
/// pixels in this frame) for each draw the scene asks for in both frames,
/// that covers pixels in both frames of `reference` (nothing refused), but
/// covers pixels in only one of the two frames of `pictures`.
fn transitions(frames: &[Frame], pictures: &[Vec<u32>], reference: &[Vec<u32>]) -> Vec<(usize, usize, u32, u32)> {
    let mut found = Vec::new();
    for n in 1..frames.len() {
        let (was, now) = (coverage(&pictures[n - 1]), coverage(&pictures[n]));
        let (ref_was, ref_now) = (coverage(&reference[n - 1]), coverage(&reference[n]));
        for r in 0..RECORDS {
            let asked = frames[n - 1].main[r].1 > 0 && frames[n].main[r].1 > 0;
            if asked && ref_was[r] > 0 && ref_now[r] > 0 && ((was[r] == 0) != (now[r] == 0)) {
                found.push((n, r, was[r], now[r]));
            }
        }
    }
    found
}

fn run(warp: &mut WarpHarness, scene: &Scene, clamp: &Clamp) -> (Vec<Frame>, Vec<Vec<u32>>) {
    let base = earlier_base();
    let frames: Vec<Frame> = (0..FRAMES).map(|f| frame(f, base)).collect();
    let pictures = frames.iter().map(|f| render(warp, scene, clamp, f)).collect();
    (frames, pictures)
}

/// The scene with a budget no frame reaches (the reference), at the
/// runtime's default 2^27 (explicitly, as a control), and through the
/// runtime-facing constructor (`make_from`: `VICTORY_INDIRECT_BUDGET_LOG2`
/// or the default). At 2^27 whole draws vanish, and some draw drawn in both
/// frames of a pair in the reference disappears or reappears between them.
/// With a budget of 2^28 or more the pictures are the reference's. The
/// findings are printed (`--nocapture`); `belobug_foliage_test.py` requires
/// the closing `foliage: completed` line. The rest-off rule at 2^27 is
/// measured too: it loses fewer draws but they change more between frames.
#[test]
fn warp_foliage_frames_keep_every_draw_asked_for() {
    let mut warp = WarpHarness::new().expect("the foliage regression needs the WARP software adapter");
    let scene = Scene::new(&warp);
    let base = earlier_base();
    let totals: Vec<u64> = (0..FRAMES)
        .map(|n| {
            let f = frame(n, base);
            cost(&f.earlier) + cost(&f.main)
        })
        .collect();
    eprintln!("foliage: list work per frame (vertex instances): {totals:?}");
    // Each record stays under the per-record bound, so only the list's
    // budget can empty a legitimate draw here; and no frame reaches 2^28.
    for n in 0..FRAMES {
        let f = frame(n, base);
        assert!(f.earlier.iter().chain(&f.main).all(|&(v, i)| u64::from(v) * u64::from(i) <= 1 << 26));
        assert!(totals[n] > 1 << 27 && totals[n] < 1 << 28);
    }

    let unbounded = make_with(&warp.device, SOURCE, 31, true).expect("clamp at 2^31");
    let (frames, reference) = run(&mut warp, &scene, &unbounded);
    let r = compare(&frames, &reference);
    eprintln!("foliage: reference 2^31: vanished {:?}; aligned differences (changed, compared) {:?}", r.vanished, r.aligned);
    assert!(r.vanished.is_empty(), "with nothing refused every draw asked for covers pixels: {:?}", r.vanished);

    // Belobug's other rest rule at the default budget: a record that does
    // not fit runs empty alone, and later smaller ones that fit still run.
    let alone = make_with(&warp.device, SOURCE, 27, false).expect("clamp at 2^27, rest off");
    let (_, skipped) = run(&mut warp, &scene, &alone);
    let s = compare(&frames, &skipped);
    let s_flicker = transitions(&frames, &skipped, &reference);
    let s_lost: usize = s.vanished.iter().map(|(_, g)| g.len()).sum();
    eprintln!("foliage: 2^27 rest off: vanished {:?}; transitions {s_flicker:?}; aligned differences {:?}", s.vanished, s.aligned);

    let raised = make_with(&warp.device, SOURCE, 28, true).expect("clamp at 2^28");
    let (_, roomy) = run(&mut warp, &scene, &raised);
    assert!(roomy == reference, "at 2^28 nothing in the scene is refused, so every frame is the reference's");

    let legacy = make_with(&warp.device, SOURCE, 27, true).expect("clamp at 2^27");
    let (_, lossy) = run(&mut warp, &scene, &legacy);
    let l = compare(&frames, &lossy);
    let flicker = transitions(&frames, &lossy, &reference);
    eprintln!("foliage: control 2^27: vanished {:?}; aligned differences {:?}", l.vanished, l.aligned);
    eprintln!("foliage: control 2^27: (frame, draw, pixels before, pixels after) {flicker:?}");
    assert!(!l.vanished.is_empty(), "the scene must reproduce whole-draw loss at 2^27");
    assert!(!flicker.is_empty(), "some draw visible in both reference frames must flicker at 2^27");
    // The draws lost are always the tail of the main pass (a record that
    // does not fit takes the rest of the budget).
    for (n, gone) in &l.vanished {
        let asked: Vec<usize> = (0..RECORDS).filter(|&r| frames[*n].main[r].1 > 0).collect();
        assert_eq!(gone[..], asked[asked.len() - gone.len()..], "frame {n}");
    }
    // The cut point itself is deterministic: a lossy frame again is the
    // same picture.
    let (n, _) = l.vanished[0];
    assert_eq!(render(&mut warp, &scene, &legacy, &frames[n]), lossy[n], "frame {n} at 2^27 again");

    let l_lost: usize = l.vanished.iter().map(|(_, g)| g.len()).sum();
    eprintln!("foliage: draws lost over {FRAMES} frames: 2^27 rest on {l_lost}, 2^27 rest off {s_lost}, 2^28 0");
    // Without the rest rule no fewer draws survive than with it.
    assert!(s_lost <= l_lost, "rest off must not lose more draws ({s_lost} > {l_lost})");

    let (log2, rest_on) = (budget_log2(), rest() != 0);
    let runtime = make_from(&warp.device, SOURCE).expect("the runtime's clamp");
    let (_, pictures) = run(&mut warp, &scene, &runtime);
    let c = compare(&frames, &pictures);
    let flicker = transitions(&frames, &pictures, &reference);
    eprintln!(
        "foliage: runtime budget 2^{log2} rest {}: vanished {:?}; transitions {flicker:?}; aligned differences {:?}",
        if rest_on { "on" } else { "off" },
        c.vanished,
        c.aligned
    );
    if log2 <= 27 {
        let control = if rest_on { &lossy } else { &skipped };
        assert!(&pictures == control, "at 2^{log2} the runtime clamp must match its 2^27 control");
    } else {
        assert!(c.vanished.is_empty() && flicker.is_empty());
        assert!(pictures == reference, "at 2^{log2} nothing is refused");
    }
    // Fixed camera: the same frame twice is the same picture.
    let still = &frames[5];
    let (a, b) = (render(&mut warp, &scene, &runtime, still), render(&mut warp, &scene, &runtime, still));
    assert!(a == b && a == pictures[5], "a still camera must give the same picture");
    eprintln!("foliage: completed at runtime budget 2^{log2}");
}

/// The single clamp and the batched clamp each admit as many records of
/// the largest draw (2^26 vertex instances) as their budget holds, at the
/// lowest and highest budgets `VICTORY_INDIRECT_BUDGET_LOG2` accepts and
/// around the default; the budget word ends full.
#[test]
fn warp_both_clamps_admit_what_the_selected_budget_holds() {
    const N: u64 = 40;
    const RECORD: [u32; 4] = [1 << 22, 16, 5, 6];
    let mut warp = WarpHarness::new().expect("the budget test needs the WARP software adapter");
    for log2 in [24u32, 27, 28, 31] {
        let fit = ((1u64 << log2) >> 26).min(N) as usize;
        for batched in [false, true] {
            let clamp = make_with(&warp.device, if batched { batch::SOURCE } else { SOURCE }, log2, true).expect("clamp");
            let slot = if batched { u64::from(batch::SLOT) } else { 16 };
            let (src, src_ptr, src_gpu) = warp.shared_buffer(N * 16);
            let (_budget, budget_ptr, budget_gpu) = warp.shared_buffer(256);
            let (back, back_ptr, _) = warp.shared_buffer(N * slot);
            let (_table, table_ptr, table_gpu) = warp.shared_buffer(batch::TABLE);
            let dst = warp.default_uav(64 << 10);
            // SAFETY: mapped buffers of these sizes, which the list has not
            // read yet.
            unsafe {
                std::ptr::write_bytes(budget_ptr, 0, 256);
                for r in 0..N as usize {
                    for (w, word) in RECORD.into_iter().enumerate() {
                        src_ptr.cast::<u32>().add(r * 4 + w).write_volatile(word);
                    }
                }
            }
            if batched {
                batch::record(&clamp, &warp.list, (&dst, D3D12_RESOURCE_STATE_COMMON), budget_gpu, [src_gpu; 4], table_gpu);
                // SAFETY: the table's mapping, which the list reads when it runs.
                let mut table = unsafe { batch::Table::new(table_ptr, 0, [1, 0, 0, 0]) };
                for r in 0..N {
                    assert_eq!(table.place(1, r * 16, 0), Some(r * slot));
                }
            } else {
                assert!(record_clamp_as(&clamp, &warp.list, (&dst, D3D12_RESOURCE_STATE_COMMON), budget_gpu, (&src, 0), None, (0, 16, N as u32), false));
            }
            // SAFETY: an open list; the records are readable as arguments.
            unsafe {
                warp.list.ResourceBarrier(&[transition(&dst, D3D12_RESOURCE_STATE_INDIRECT_ARGUMENT, D3D12_RESOURCE_STATE_COPY_SOURCE)]);
                warp.list.CopyBufferRegion(&back, 0, &dst, 0, N * slot);
            }
            warp.submit_and_wait();
            // SAFETY: the list has run; mapped words.
            let (spent, kept) = unsafe {
                let kept = (0..N as usize)
                    .filter(|&r| {
                        let at = back_ptr.cast::<u32>().add(r * slot as usize / 4);
                        let words = [at.read_volatile(), at.add(1).read_volatile()];
                        assert!(words == [RECORD[0], RECORD[1]] || words == [0, 0], "2^{log2} batched {batched}: {words:?}");
                        words[0] != 0
                    })
                    .count();
                (budget_ptr.cast::<u32>().read_volatile(), kept)
            };
            assert_eq!((kept, spent), (fit, 1u32 << log2), "2^{log2}, batched {batched}");
        }
    }
}

#[test]
fn budget_log2_reads_only_values_the_clamp_can_sum() {
    assert_eq!(DEFAULT_BUDGET_LOG2, 27, "the default stays the pinned runtime's");
    assert_eq!(parse_budget_log2(None), DEFAULT_BUDGET_LOG2);
    assert_eq!(parse_budget_log2(Some("24")), 24);
    assert_eq!(parse_budget_log2(Some("28")), 28);
    assert_eq!(parse_budget_log2(Some("31")), 31);
    for bad in ["", "x", "23", "32", "-1", "28 "] {
        assert_eq!(parse_budget_log2(Some(bad)), DEFAULT_BUDGET_LOG2, "{bad:?}");
    }
}

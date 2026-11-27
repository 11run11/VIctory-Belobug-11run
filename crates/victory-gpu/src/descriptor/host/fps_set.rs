//! Original code written for the XboxCompatibility project (task T-0041), not
//! copied from any other tree. It follows two ideas described in Xeltax/VIcky's
//! notes (VICKY-RT.md, switches `VICTORY_HEAP_NOTE_KIND` and
//! `VICTORY_HEAP_SYNC_LEAN`), re-thought for this tree's heap code.
//! SPDX-License-Identifier: GPL-3.0-or-later
//!
//! Both switches are opt-in (off unless set) and LOCAL-ONLY builds carry them.
//!
//! `VICTORY_HEAP_NOTE_KIND=1|check` ([`HostHeap::note_buffer`]): each title
//! slot remembers what its last full note left it as: [`BARE`] (no view, or
//! one that is neither a buffer's nor a texture's), [`PLAIN`] (a buffer view
//! with a stride) or [`TEXTURE`] (a placed texture or a texture object, or a
//! slot whose kind is not known). Only a texture note puts a slot in the
//! queue of textures picked by index or in the placed spans. A note from
//! bare or plain to bare or plain therefore skips the queue's look, the
//! spans' look and their counters, and does only the stride table and the
//! buffer sets. `check`: every note is a full one, as without the switch,
//! and the notes that would have been short count how often the slot's
//! queue or span bit was set (where the short way would have left it).
//!
//! `VICTORY_HEAP_SYNC_LEAN=1|check` (`super::super::sync_heaps`): the slots
//! the heap sync makes are made from the record its compare step already
//! holds (the page copy, or the records last seen), not from a second read
//! of title memory. A record the title writes again in between differs from
//! the one now last seen, so the next sync makes it. `check`: made from the
//! title's record as before, counting those whose record differed from the
//! compare step's copy.
//!
//! Xeltax/VIcky's third measured fps switch, `VICTORY_TEXTURE_PASS_CHUNK`,
//! is not ported: it batches one global lock of made slots per chunk, and
//! this tree's made-slot lock is already striped 512 ways (`Made::lock`).

use std::sync::OnceLock;
use std::sync::atomic::{AtomicU8, AtomicU64, Ordering::Relaxed};

/// No view, or a view that is neither a buffer's nor a texture's.
pub(crate) const BARE: u8 = 0;
/// A buffer view with an element stride.
pub(crate) const PLAIN: u8 = 1;
/// A texture's view, or not known: the next note is a full one.
pub(crate) const TEXTURE: u8 = 2;

/// What a switch asks for.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Mode {
    Off,
    On,
    Check,
}

/// The value of a switch variable as a mode (anything else: off).
pub(crate) fn parse(value: Option<&str>) -> Mode {
    match value {
        Some("1") | Some("on") => Mode::On,
        Some("check") => Mode::Check,
        _ => Mode::Off,
    }
}

fn read(name: &str) -> Mode {
    let mode = parse(std::env::var(name).ok().as_deref());
    if mode != Mode::Off {
        tracing::info!("{name}={mode:?} (T-0041 fps set, opt-in)");
    }
    mode
}

/// `VICTORY_HEAP_NOTE_KIND`, read once.
pub(crate) fn note_kind() -> Mode {
    static MODE: OnceLock<Mode> = OnceLock::new();
    *MODE.get_or_init(|| read("VICTORY_HEAP_NOTE_KIND"))
}

/// `VICTORY_HEAP_SYNC_LEAN`, read once.
pub(crate) fn sync_lean() -> Mode {
    static MODE: OnceLock<Mode> = OnceLock::new();
    *MODE.get_or_init(|| read("VICTORY_HEAP_SYNC_LEAN"))
}

/// Whether a note from kind `was` to kind `now` may skip the queue and spans.
pub(crate) fn short(was: u8, now: u8) -> bool {
    was != TEXTURE && now != TEXTURE
}

/// Since the last report: short notes; check-mode notes that would have been
/// short; of those, ones whose queue or span bit was set; lean makes; lean
/// check makes; of those, ones whose record in title memory differed.
pub(crate) static COUNTS: [AtomicU64; 6] = [const { AtomicU64::new(0) }; 6];

pub(crate) fn count(which: usize) {
    COUNTS[which].fetch_add(1, Relaxed);
}

/// The kind each title slot's last full note left it as.
pub(crate) struct Kinds(Box<[AtomicU8]>);

impl Kinds {
    /// All slots start as [`TEXTURE`] (not known): each one's first note is
    /// a full one.
    pub(crate) fn new(slots: usize) -> Kinds {
        Kinds((0..slots).map(|_| AtomicU8::new(TEXTURE)).collect())
    }

    pub(crate) fn get(&self, index: u32) -> u8 {
        self.0.get(index as usize).map_or(TEXTURE, |k| k.load(Relaxed))
    }

    /// (Under the slot's lock, as every note.)
    pub(crate) fn set(&self, index: u32, kind: u8) {
        if let Some(k) = self.0.get(index as usize) {
            k.store(kind, Relaxed);
        }
    }
}

/// One info line per report while either switch is on.
pub(crate) fn line(frames: f64) {
    if note_kind() == Mode::Off && sync_lean() == Mode::Off {
        return;
    }
    let c = [0, 1, 2, 3, 4, 5].map(|i| COUNTS[i].swap(0, Relaxed));
    tracing::info!(
        "fps set per frame: VICTORY_HEAP_NOTE_KIND={:?} {:.0} short notes, check {:.0} would be short / {} with a queue or span bit; \
         VICTORY_HEAP_SYNC_LEAN={:?} {:.0} slots made from the compared record, check {:.0} / {} whose record had changed",
        note_kind(),
        c[0] as f64 / frames,
        c[1] as f64 / frames,
        c[2],
        sync_lean(),
        c[3] as f64 / frames,
        c[4] as f64 / frames,
        c[5]
    );
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn switches_are_opt_in() {
        assert_eq!(parse(None), Mode::Off);
        assert_eq!(parse(Some("0")), Mode::Off);
        assert_eq!(parse(Some("yes")), Mode::Off);
        assert_eq!(parse(Some("1")), Mode::On);
        assert_eq!(parse(Some("check")), Mode::Check);
    }

    #[test]
    fn only_non_texture_notes_are_short() {
        for (was, now, ok) in [
            (BARE, BARE, true),
            (BARE, PLAIN, true),
            (PLAIN, BARE, true),
            (PLAIN, PLAIN, true),
            (TEXTURE, PLAIN, false),
            (PLAIN, TEXTURE, false),
            (TEXTURE, TEXTURE, false),
        ] {
            assert_eq!(short(was, now), ok, "{was} -> {now}");
        }
    }

    #[test]
    fn unknown_slots_take_the_full_note() {
        let kinds = Kinds::new(4);
        assert_eq!(kinds.get(2), TEXTURE);
        kinds.set(2, PLAIN);
        assert_eq!(kinds.get(2), PLAIN);
        assert_eq!(kinds.get(99), TEXTURE);
        kinds.set(99, PLAIN);
        assert_eq!(kinds.get(99), TEXTURE);
    }
}

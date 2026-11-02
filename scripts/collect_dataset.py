"""Perekam dataset postur multi-sesi dari telemetry MQTT Edge 1 dengan validasi anti-stale & anti-replay.

Contoh penggunaan:
  python -m scripts.collect_dataset --label POSISI_DUDUK_SEHAT --samples 40
  python -m scripts.collect_dataset --label DITARUH_TELUNGKUP --samples 40
  python -m scripts.collect_dataset --label REBAHAN_AKTIF --samples 40
"""

import argparse
from collections import defaultdict
import csv
from datetime import datetime
from pathlib import Path
import time
import uuid

from shared.messages import Telemetry
from shared.mqtt_connection import create_client, topic
from shared.processing import safe_lean_deg

VALID_LABELS = ("POSISI_DUDUK_SEHAT", "DITARUH_TELUNGKUP", "REBAHAN_AKTIF")
CSV_COLUMNS = [
    "session_id",
    "timestamp_ms",
    "overhead_deg",
    "roll_deg",
    "lean_deg",
    "motion_rms",
    "gyro_rms",
    "label",
]
STALE_MESSAGE_MS = 30_000


def generate_session_id(now: datetime | None = None) -> str:
    dt = now or datetime.now()
    return f"sess_{dt.strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:4]}"


class DatasetCollector:
    """Memvalidasi paket Telemetry Edge 1 (anti-stale, anti-replay, khusus edge1) dan menyimpan ke CSV."""

    def __init__(
        self,
        label: str,
        output_path: str | Path = "data/posture_dataset.csv",
        target_samples: int = 40,
        session_id: str | None = None,
        stale_message_ms: int = STALE_MESSAGE_MS,
    ) -> None:
        if label not in VALID_LABELS:
            raise ValueError(f"Label '{label}' tidak valid. Pilih salah satu dari: {', '.join(VALID_LABELS)}")
        if target_samples <= 0:
            raise ValueError("Jumlah --samples harus > 0")

        self.label = label
        self.output_path = Path(output_path)
        self.target_samples = target_samples
        self.session_id = (session_id or "").strip() or generate_session_id()
        self.stale_message_ms = stale_message_ms

        self.last_seq: dict[tuple[str, str], int] = {}
        self.collected_count: int = 0
        self.rejected_stale: int = 0
        self.rejected_replay: int = 0
        self.rejected_device: int = 0
        self.rejected_invalid: int = 0

        self._ensure_csv_header()

    def _ensure_csv_header(self) -> None:
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        if not self.output_path.exists() or self.output_path.stat().st_size == 0:
            with self.output_path.open("w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow(CSV_COLUMNS)

    def accept_payload(
        self, raw_payload: str | bytes, now_ms: int | None = None
    ) -> tuple[bool, str, dict[str, object] | None]:
        """Memvalidasi satu payload JSON Telemetry dan menulis ke CSV jika valid."""
        if self.collected_count >= self.target_samples:
            return False, "QUOTA_REACHED", None

        try:
            data = Telemetry.from_json(raw_payload)
        except ValueError as exc:
            self.rejected_invalid += 1
            return False, f"REJECTED_INVALID: {exc}", None

        if data.device_id != "edge1":
            self.rejected_device += 1
            return False, "REJECTED_DEVICE", None

        current_ms = now_ms if now_ms is not None else int(time.time() * 1000)
        if abs(current_ms - data.sent_at_ms) > self.stale_message_ms:
            self.rejected_stale += 1
            return False, "REJECTED_STALE", None

        key = (data.device_id, data.boot_id)
        if data.sequence <= self.last_seq.get(key, -1):
            self.rejected_replay += 1
            return False, "REJECTED_REPLAY", None
        self.last_seq[key] = data.sequence

        lean_deg = safe_lean_deg(data.overhead_deg, data.roll_deg)
        row = {
            "session_id": self.session_id,
            "timestamp_ms": data.sent_at_ms,
            "overhead_deg": round(float(data.overhead_deg), 4),
            "roll_deg": round(float(data.roll_deg), 4),
            "lean_deg": round(float(lean_deg), 4),
            "motion_rms": round(float(data.motion_rms), 4),
            "gyro_rms": round(float(data.gyro_rms), 4),
            "label": self.label,
        }

        with self.output_path.open("a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS)
            writer.writerow(row)

        self.collected_count += 1
        return True, "ACCEPTED", row


def summarize_dataset(csv_path: str | Path) -> dict[str, dict[str, int]]:
    path = Path(csv_path)
    counts: dict[str, int] = defaultdict(int)
    sessions: dict[str, set[str]] = defaultdict(set)
    if path.exists():
        with path.open("r", newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                lbl = (row.get("label") or "").strip()
                sid = (row.get("session_id") or "").strip()
                if lbl:
                    counts[lbl] += 1
                    if sid:
                        sessions[lbl].add(sid)

    summary: dict[str, dict[str, int]] = {}
    for lbl in VALID_LABELS:
        summary[lbl] = {
            "samples": counts[lbl],
            "sessions": len(sessions[lbl]),
        }
    return summary


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="RebahGuard — Perekam Dataset Telemetry Edge 1 Multi-Sesi"
    )
    parser.add_argument(
        "--label",
        required=True,
        choices=VALID_LABELS,
        help="Label postur yang sedang direkam",
    )
    parser.add_argument(
        "--samples",
        type=int,
        default=40,
        help="Jumlah window sampel yang ingin direkam pada sesi ini (default: 40)",
    )
    parser.add_argument(
        "--session-id",
        default="",
        help="ID sesi opsional (otomatis sess_<YYYYMMDD_HHMMSS>_<4hex> jika kosong)",
    )
    parser.add_argument(
        "--output",
        default="data/posture_dataset.csv",
        help="Path file CSV tujuan (default: data/posture_dataset.csv)",
    )
    parser.add_argument(
        "--countdown",
        type=int,
        default=3,
        help="Hitung mundur detik sebelum perekaman dimulai (default: 3)",
    )
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_arg_parser().parse_args(argv)
    collector = DatasetCollector(
        label=args.label,
        output_path=args.output,
        target_samples=args.samples,
        session_id=args.session_id or None,
    )

    print("=== REBAHGUARD DATASET COLLECTOR ===")
    print(f"Label Target : {collector.label}")
    print(f"Session ID   : {collector.session_id}")
    print(f"Target Sampel: {collector.target_samples}")
    print(f"File Output  : {collector.output_path}")

    for remaining in range(max(0, args.countdown), 0, -1):
        print(f"Siapkan posisi HP Anak ({collector.label})... mulai dalam {remaining} detik")
        time.sleep(1.0)

    sub_topic = topic("edge/edge1/telemetry")
    client = create_client("dataset-collector")

    def on_connect(c, _userdata, _flags, reason, _properties) -> None:
        if reason.is_failure:
            print(f"[COLLECT] Gagal terhubung ke broker: {reason}")
            return
        c.subscribe(sub_topic, qos=1)
        print(f"[COLLECT] Terhubung & subscribe ke {sub_topic}. Menunggu telemetry Edge 1...")

    def on_message(c, _userdata, message) -> None:
        accepted, status, row = collector.accept_payload(message.payload)
        if accepted and row is not None:
            print(
                f"[{collector.collected_count}/{collector.target_samples}] "
                f"sess={row['session_id']} overhead={row['overhead_deg']:.1f}° "
                f"roll={row['roll_deg']:.1f}° lean={row['lean_deg']:.1f}° "
                f"motion={row['motion_rms']:.4f} gyro={row['gyro_rms']:.4f} -> {row['label']}"
            )
            if collector.collected_count >= collector.target_samples:
                print("[COLLECT] Target sampel sesi ini tercapai! Menutup koneksi...")
                c.disconnect()
        elif status in ("REJECTED_STALE", "REJECTED_REPLAY"):
            print(f"[COLLECT] Paket diabaikan ({status})")

    client.on_connect = on_connect
    client.on_message = on_message

    try:
        client.loop_forever()
    except KeyboardInterrupt:
        print("\n[COLLECT] Dihentikan oleh pengguna.")
    finally:
        try:
            client.disconnect()
        except Exception:
            pass

    summary = summarize_dataset(collector.output_path)
    print("\n=== RINGKASAN DATASET SAAT INI ===")
    for lbl, stats in summary.items():
        print(f"- {lbl:<20}: {stats['samples']:>4} sampel ({stats['sessions']} sesi unik)")


if __name__ == "__main__":
    main()


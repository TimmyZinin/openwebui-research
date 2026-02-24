#!/usr/bin/env python3
"""
AI Video Factory — Scheduler
Runs video production on a schedule for all bloggers.

Usage:
    python3 scheduler.py                    # Run once for all bloggers
    python3 scheduler.py --daemon           # Run continuously (cron-like)
    python3 scheduler.py --blogger lisa     # Run for specific blogger
    python3 scheduler.py --topic "my topic" # Override topic
"""

import os
import sys
import json
import time
import random
import logging
from pathlib import Path
from datetime import datetime, timedelta

# Setup paths
BASE_DIR = Path("/opt/ai-video")
sys.path.insert(0, str(BASE_DIR / "scripts"))
os.chdir(str(BASE_DIR))

from dotenv import load_dotenv
load_dotenv(BASE_DIR / "config" / ".env")

from video_pipeline import produce_video

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(BASE_DIR / "logs" / "scheduler.log"),
        logging.StreamHandler(),
    ],
)
log = logging.getLogger("scheduler")

# Blogger configurations
BLOGGERS = {
    "lisa": {
        "name": "Lisa",
        "topics": [
            "5 ошибок на собеседовании, которые стоят вам оффер",
            "Как правильно уволиться без скандала",
            "3 способа произвести впечатление на HR",
            "Что писать в сопроводительном письме в 2026 году",
            "Как договориться о зарплате на 30% выше",
            "7 красных флагов в вакансии — как их распознать",
            "Резюме на 1 странице: как уместить 10 лет опыта",
            "Первые 90 дней на новой работе — план выживания",
            "Как пройти стресс-интервью и не потерять самообладание",
            "Нетворкинг для интровертов: 5 работающих стратегий",
            "Как AI меняет рынок труда — что делать уже сейчас",
            "Фриланс vs офис: честное сравнение в 2026",
            "Как составить портфолио, если нет опыта",
            "Выгорание: 5 признаков и как из него выйти",
            "LinkedIn профиль: 3 изменения, которые привлекут рекрутёров",
            "Как перейти в IT без технического образования",
            "Soft skills, которые ценят работодатели больше всего",
            "Удалённая работа: как оставаться продуктивным дома",
            "Собеседование на английском: подготовка за 24 часа",
            "Карьерный план на 5 лет: пошаговая инструкция",
        ],
        "videos_per_day": 3,
        "voice": "Laura",
    },
}

# Future bloggers (placeholder)
# "ryan": {"name": "Ryan", "topics": [...], "videos_per_day": 3, "voice": "Adam"},
# "sophia": {"name": "Sophia", "topics": [...], "videos_per_day": 3, "voice": "Bella"},


def get_topic(blogger_config: dict) -> str:
    """Get a random topic for the blogger."""
    return random.choice(blogger_config["topics"])


def run_production(blogger_key: str, topic: str = None) -> Path | None:
    """Run video production for a blogger."""
    config = BLOGGERS.get(blogger_key)
    if not config:
        log.error(f"Unknown blogger: {blogger_key}")
        return None

    if not topic:
        topic = get_topic(config)

    log.info(f"Starting production: {config['name']} — {topic}")
    try:
        output = produce_video(topic, config["name"])
        if output:
            log.info(f"Video ready: {output}")
        else:
            log.warning(f"Production returned None for {config['name']}")
        return output
    except Exception as e:
        log.error(f"Production failed for {config['name']}: {e}")
        return None


def run_all_bloggers():
    """Run production for all bloggers."""
    results = {}
    for key, config in BLOGGERS.items():
        for i in range(config["videos_per_day"]):
            topic = get_topic(config)
            log.info(f"[{key}] Video {i+1}/{config['videos_per_day']}: {topic}")
            output = run_production(key, topic)
            results.setdefault(key, []).append(
                {"topic": topic, "output": str(output) if output else None}
            )
            time.sleep(5)  # Brief pause between videos

    # Save daily report
    report_path = BASE_DIR / "logs" / f"report_{datetime.now().strftime('%Y%m%d')}.json"
    report_path.write_text(json.dumps(results, ensure_ascii=False, indent=2))
    log.info(f"Daily report: {report_path}")
    return results


def daemon_mode():
    """Run continuously, producing videos on schedule."""
    log.info("Daemon mode started. Producing videos daily.")
    while True:
        now = datetime.now()
        # Produce at 8:00, 13:00, 18:00 (Moscow time = CET+2)
        # VPS is CET, so 6:00, 11:00, 16:00
        production_hours = [6, 11, 16]

        if now.hour in production_hours and now.minute < 5:
            log.info(f"Production window: {now.hour}:00")
            for key, config in BLOGGERS.items():
                topic = get_topic(config)
                run_production(key, topic)
                time.sleep(10)

        # Sleep until next check (5 minutes)
        time.sleep(300)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="AI Video Factory Scheduler")
    parser.add_argument("--daemon", action="store_true", help="Run in daemon mode")
    parser.add_argument("--blogger", type=str, help="Run for specific blogger")
    parser.add_argument("--topic", type=str, help="Override topic")
    args = parser.parse_args()

    if args.daemon:
        daemon_mode()
    elif args.blogger:
        run_production(args.blogger, args.topic)
    else:
        run_all_bloggers()

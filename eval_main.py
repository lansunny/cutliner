"""No-argument mode only validates annotation readiness. Model runs require an explicit command."""

import argparse
import asyncio
from pathlib import Path

from mini_agent.video.evaluation import coverage, load_cases, prepare, run, summarize, validate_sources


def main():
    root = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description="视频检索三组对照评测")
    parser.add_argument("command", nargs="?", choices=["validate", "prepare", "coverage", "run", "summarize"], default="validate")
    parser.add_argument("--dataset", type=Path, default=root / "evaluation_cases.json")
    parser.add_argument("--output", type=Path, default=root / "runs" / "eval_dev_001")
    parser.add_argument("--split", choices=["dev", "eval"], default="dev")
    parser.add_argument("--max-images", type=int, default=2000)
    args = parser.parse_args()
    if args.command == "validate":
        cases = load_cases(args.dataset)
        validate_sources(cases)
        labeled_count = sum(c.annotation_status == 'labeled' for c in cases)
        print(f"共{len(cases)}条任务，已人工标注{labeled_count}条。")
        if labeled_count == len(cases) and cases:
            print("标注格式与视频时长检查通过。下一步将脚本参数设为prepare，准备本地采样素材。")
        else:
            print("请填写video_path、target_exists、events、annotator，再将annotation_status设为labeled。")
            print("events使用当前素材内的相对秒数；不存在目标时events=[]、target_exists=false、category=negative。")
        print("无模型调用。标注文件：", args.dataset)
    elif args.command == "prepare":
        prepare(args.dataset, args.output, args.split)
    elif args.command == "run":
        if args.max_images < 1:
            parser.error("--max-images必须为正整数")
        print("开始付费API评测；计数包括粗扫描、补看和成片复核中的重复图片输入。")
        asyncio.run(run(args.output, args.max_images))
    elif args.command == "coverage":
        coverage(args.output)
    else:
        summarize(args.output)


if __name__ == "__main__":
    try:
        main()
    except (ValueError, OSError) as error:
        print("评测未继续：", error)
        raise SystemExit(2)

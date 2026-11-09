import json
from datetime import datetime
from pathlib import Path
from uuid import uuid4

from mini_agent.config import PROJECT_ROOT
from mini_agent.video.reader import VideoReader
from mini_agent.video.shot_sampler import ShotFrameSampler


def main() -> None:
    video_path = input("请输入用于镜头切分的本地视频路径：").strip().strip('"')
    if not video_path:
        raise ValueError("视频路径不能为空")
    metadata = VideoReader().read_metadata(Path(video_path))
    run_dir = PROJECT_ROOT / "runs" / f"shots_{datetime.now():%Y%m%d_%H%M%S}_{uuid4().hex[:8]}"
    run_dir.mkdir(parents=True, exist_ok=False)
    print("结果目录：", run_dir)
    result = ShotFrameSampler().prepare(metadata, run_dir / "frames")
    data = {
        "metadata.json": metadata.model_dump(mode="json"),
        "shots.json": [shot.model_dump(mode="json") for shot in result.shots],
        "frames.json": [frame.model_dump(mode="json") for frame in result.frames],
        "sampling_stats.json": {
            "strategy": "hsv_histogram_peaks",
            "config": result.config.model_dump(mode="json"),
            "stats": result.stats.model_dump(mode="json"),
        },
    }
    for filename, content in data.items():
        (run_dir / filename).write_text(json.dumps(content, ensure_ascii=False, indent=2), encoding="utf-8")
    print("\n========== 镜头切分与采样结果 ==========")
    for shot in result.shots:
        print(f"{shot.shot_id}：{shot.start_seconds:.3f}～{shot.end_seconds:.3f}秒，代表帧{len(shot.representative_frame_indices)}张")
    stats = result.stats
    print(f"原始{stats.original_frame_count}帧；缩略图扫描{stats.scan_frame_count}帧；代表帧{stats.representative_frame_count}张。")
    if stats.reduction_vs_uniform >= 0:
        print(f"每2秒均匀采样需{stats.uniform_two_second_frame_count}张，本次相对减少{stats.reduction_vs_uniform:.1%}。")
    else:
        print(f"每2秒均匀采样仅{stats.uniform_two_second_frame_count}张，本次镜头代表帧更多，以保留各镜头覆盖。")
    print(f"相对原始帧减少{stats.reduction_vs_original:.1%}；处理耗时{stats.elapsed_seconds:.2f}秒。")
    print("本次没有调用模型。完整视频入口也可输入这个结果目录，复用这些代表帧。")


if __name__ == "__main__":
    main()

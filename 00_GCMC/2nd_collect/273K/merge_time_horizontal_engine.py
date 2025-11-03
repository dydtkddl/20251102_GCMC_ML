import pandas as pd
import re
import os
import argparse
from functools import reduce

def merge_time_data_horizontal(T2: int):
    """
    TIME CSV들을 MOF 기준으로 병합해
    HENRY~15bar까지 열로 갖는 horizontal pivot 생성

    예: merge_time_data_horizontal(293) → ./293K_time_horizontal.csv
    """
    T = f"{T2}K"
    base_path = f"./{T}/TIME"

    PATHS = {
        "HENRY": f"{base_path}/{T2}_HENRY.csv",
        "0.01": f"{base_path}/{T2}_0.01.csv",
        "0.05": f"{base_path}/{T2}_0.05.csv",
        "0.1":  f"{base_path}/{T2}_0.1.csv",
        "0.5":  f"{base_path}/{T2}_0.5.csv",
        "1":    f"{base_path}/{T2}_1.csv",
        "5":    f"{base_path}/{T2}_5.csv",
        "15":   f"{base_path}/{T2}_15.csv"
    }

    dfs = []
    print(f"\n🚀 [{T}] TIME 데이터 병합 시작")
    print("=" * 80)

    for label, path in PATHS.items():
        if not os.path.exists(path):
            print(f"⚠️  Skip (파일 없음): {path}")
            continue

        df = pd.read_csv(path)
        df.columns = [c.strip().lower().replace(" ", "_").replace("(s)", "s") for c in df.columns]

        # name / time 컬럼 자동 탐색
        name_candidates = [c for c in df.columns if "mof" in c or "name" in c]
        time_candidates = [c for c in df.columns if "time" in c]

        if not name_candidates or not time_candidates:
            print(f"⚠️  {path} → name/time 컬럼 탐지 실패 (건너뜀)")
            continue

        name_col = name_candidates[0]
        time_col = time_candidates[0]

        # name 정제
        def clean_name(name):
            return re.sub(r"_\d+(?:\.\d+)?bar.*", "", str(name))

        df["name_base"] = df[name_col].apply(clean_name)
        df = df[["name_base", time_col]].rename(columns={time_col: label})
        dfs.append(df)

    if not dfs:
        raise RuntimeError(f"❌ {T}에서 병합할 유효한 TIME CSV 없음")

    merged_df = reduce(lambda left, right: pd.merge(left, right, on="name_base", how="outer"), dfs)
    order = ["name_base", "HENRY", "0.01", "0.05", "0.1", "0.5", "1", "5", "15"]
    merged_df = merged_df[[c for c in order if c in merged_df.columns]]

    save_path = f"./{T}_time_horizontal.csv"
    merged_df.to_csv(save_path, index=False)
    print(f"✅ Horizontal 병합 완료 → {save_path}")
    print(f"📦 총 {len(merged_df):,}개 MOF 변환됨")
    print("=" * 80)

    return merged_df


def main():
    parser = argparse.ArgumentParser(
        description="TIME CSV 병합 엔진 (Henry~15bar horizontal pivot 생성)"
    )
    parser.add_argument(
        "--temps",
        nargs="+",
        type=int,
        required=True,
        help="병합할 온도 리스트 (예: 273 293 313)"
    )
    args = parser.parse_args()

    for T2 in args.temps:
        try:
            merge_time_data_horizontal(T2)
        except Exception as e:
            print(f"❌ {T2}K 처리 중 오류 발생: {e}")


if __name__ == "__main__":
    main()

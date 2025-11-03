import os
import argparse
import pandas as pd
from functools import reduce


def merge_uptake_data(temp):
    """
    주어진 온도(K)에 대해 UPTAKE 폴더의 흡착 데이터를 병합.
    HENRY는 henry_coeff, 나머지는 abs_mol_per_kg_framework 기준.
    """
    T = f"{temp}K"
    base_dir = f"./{T}/UPTAKE"
    if not os.path.exists(base_dir):
        raise FileNotFoundError(f"❌ UPTAKE 폴더가 존재하지 않습니다: {base_dir}")

    # 병합 대상 파일 정의
    paths = {
        "HENRY": f"{base_dir}/{temp}_HENRY.csv",
        "0.01": f"{base_dir}/{temp}_0.01.csv",
        "0.05": f"{base_dir}/{temp}_0.05.csv",
        "0.1":  f"{base_dir}/{temp}_0.1.csv",
        "0.5":  f"{base_dir}/{temp}_0.5.csv",
        "1":    f"{base_dir}/{temp}_1.csv",
        "5":    f"{base_dir}/{temp}_5.csv",
        "15":   f"{base_dir}/{temp}_15.csv"
    }

    dfs = []
    for label, path in paths.items():
        if not os.path.exists(path):
            print(f"⚠️ 경고: {path} 파일이 존재하지 않음 (건너뜀)")
            continue

        df = pd.read_csv(path)
        if label == "HENRY":
            value_col = "henry_coeff"
        else:
            value_col = "abs_mol_per_kg_framework"

        if value_col not in df.columns:
            raise KeyError(f"'{path}' 파일에 '{value_col}' 컬럼이 없습니다.")

        df = df[["name", value_col]].rename(columns={value_col: label})
        dfs.append(df)

    if not dfs:
        raise ValueError(f"❌ {T}에 대해 병합할 수 있는 파일이 없습니다.")

    merged = reduce(lambda left, right: pd.merge(left, right, on="name", how="outer"), dfs)
    cols_order = ["name"] + list(paths.keys())
    merged = merged.reindex(columns=cols_order)

    out_path = f"./{T}_uptake_pivot.csv"
    merged.to_csv(out_path, index=False)

    print("=" * 70)
    print(f"📊 [{T}] Uptake 파일 병합 완료")
    print(f" - 총 구조 수: {len(merged):,}")
    print(f" - 출력 파일: {out_path}")
    print("=" * 70)
    return out_path


def main():
    parser = argparse.ArgumentParser(
        description="다중 온도 Uptake 데이터 병합 엔진 (Henry + Uptake)"
    )
    parser.add_argument(
        "--temps",
        nargs="+",
        type=int,
        required=True,
        help="병합할 온도 리스트 (예: 273 293 313)",
    )
    args = parser.parse_args()

    for temp in args.temps:
        try:
            merge_uptake_data(temp)
        except Exception as e:
            print(f"❌ {temp}K 처리 중 오류 발생: {e}")


if __name__ == "__main__":
    main()

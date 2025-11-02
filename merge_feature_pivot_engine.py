import argparse
import pandas as pd
import os

def merge_feature_and_pivot_left(feature_file, pivot_file, output_file):
    """
    Feature CSV를 왼쪽 기준으로, Pivot CSV를 오른쪽 기준으로 병합.
    filename ↔ name 기준 inner join, pivot의 name 컬럼은 제거 후 저장.
    """
    # ─────────────── 파일 확인 ───────────────
    if not os.path.exists(feature_file):
        raise FileNotFoundError(f"❌ Feature 파일을 찾을 수 없습니다: {feature_file}")
    if not os.path.exists(pivot_file):
        raise FileNotFoundError(f"❌ Pivot 파일을 찾을 수 없습니다: {pivot_file}")

    # ─────────────── CSV 로드 ───────────────
    print(f"📂 Feature 파일 로드 중: {feature_file}")
    feature_df = pd.read_csv(feature_file)
    print(f"📂 Pivot 파일 로드 중: {pivot_file}")
    pivot_df = pd.read_csv(pivot_file)

    # ─────────────── Feature 컬럼 선택 ───────────────
    feature_cols = [
        "filename", "LCD", "PLD", "LFPD", "cm3_g",
        "ASA_m2_cm3", "ASA_m2_g", "NASA_m2_cm3", "NASA_m2_g",
        "AV_VF", "AV_cm3_g", "NAV_cm3_g", "Has_OMS"
    ]
    feature_df = feature_df[feature_cols].copy()

    # ─────────────── Has_OMS Label Encoding ───────────────
    feature_df["Has_OMS"] = (
        feature_df["Has_OMS"].astype(str).str.strip().str.lower().map({"yes": 1, "no": 0})
    )
    feature_df["Has_OMS"].fillna(0, inplace=True)

    # ─────────────── 병합 ───────────────
    merged = pd.merge(
        feature_df,
        pivot_df,
        left_on="filename",
        right_on="name",
        how="inner"
    )

    # ─────────────── name 컬럼 제거 ───────────────
    if "name" in merged.columns:
        merged.drop(columns=["name"], inplace=True)

    print("\n✅ 병합 완료!")
    print(f"📊 Feature shape: {feature_df.shape}")
    print(f"📊 Pivot shape: {pivot_df.shape}")
    print(f"📊 병합 결과 shape: {merged.shape}")

    # ─────────────── 저장 ───────────────
    merged.to_csv(output_file, index=False)
    print(f"\n💾 결과 저장 완료 → {output_file}")


def main():
    parser = argparse.ArgumentParser(
        description="Feature(왼쪽) + Pivot(오른쪽) inner join 병합 엔진"
    )
    parser.add_argument(
        "--input_feature_base_file",
        required=True,
        help="왼쪽 Feature CSV 파일 경로 (예: 2019-11-01-ASR-public_12020.csv)",
    )
    parser.add_argument(
        "--pivot_file",
        required=True,
        help="오른쪽 Uptake Pivot CSV 파일 경로 (예: 313K_uptake_pivot.csv)",
    )
    parser.add_argument(
        "--output_file",
        default="./merged_dataset.csv",
        help="병합 결과 저장 파일명 (기본값: merged_dataset.csv)",
    )

    args = parser.parse_args()
    merge_feature_and_pivot_left(args.input_feature_base_file, args.pivot_file, args.output_file)


if __name__ == "__main__":
    main()




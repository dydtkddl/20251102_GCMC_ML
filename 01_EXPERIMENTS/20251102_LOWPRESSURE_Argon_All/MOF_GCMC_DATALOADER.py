"""
MOF_GCMC_DATALOADER.py
────────────────────────────────────────────
기능:
 - CSV 파일을 읽고,
 - 인풋 피처, 저압(로우프레셔) 피처, 아웃풋 피처를 선택한 뒤
 - 필요 시 '다른 모델의 예측 컬럼(pred_features)'을 추가 입력으로 포함
 - 메타데이터를 포함한 통합 DataFrame 반환

사용 예시:
from MOF_GCMC_DATALOADER import load_mof_dataset

df, meta = load_mof_dataset(
    csv_path="./merged_dataset.csv",
    input_features=["LCD", "PLD", "LFPD", "cm3_g", "ASA_m2_g", "Has_OMS"],
    lowp_features=["HENRY", "0.01", "0.05"],
    output_features=["0.1", "0.5", "1", "5", "15"],
    pred_features=["pred_0.1", "pred_0.5", "pred_1"]  # ← 다른 모델 예측값 포함
)
"""

import pandas as pd
import os


def load_mof_dataset(csv_path: str,
                     input_features: list,
                     lowp_features: list,
                     output_features: list,
                     meta_cols: list = None,
                     pred_features: list = None,
                     dropna: bool = True):
    """
    CSV를 읽고 피처를 정리한 DataFrame과 메타정보를 반환

    Parameters
    ----------
    csv_path : str
        읽을 CSV 파일 경로
    input_features : list[str]
        기본 입력 피처 (e.g., ["LCD", "PLD", "cm3_g", ...])
    lowp_features : list[str]
        저압 입력 피처 (e.g., ["HENRY", "0.01", "0.05"])
    output_features : list[str]
        출력 피처 (e.g., ["0.1", "0.5", "1", "5", "15"])
    meta_cols : list[str], optional
        함께 포함할 메타데이터 컬럼 (기본값: ["filename"])
    pred_features : list[str], optional
        다른 모델의 예측값 컬럼 (기본값: None)
    dropna : bool, optional
        결측치 행 제거 여부 (기본값: True)

    Returns
    -------
    df_final : pd.DataFrame
        정리된 DataFrame
    meta : dict
        데이터셋 메타정보 요약
    """

    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"❌ 파일을 찾을 수 없습니다: {csv_path}")

    df = pd.read_csv(csv_path)
    print(f"📂 CSV 로드 완료: {csv_path} ({len(df):,}행)")

    # 기본 메타 컬럼
    if meta_cols is None:
        meta_cols = ["filename"]

    # 예측 컬럼이 없는 경우 빈 리스트로 처리
    if pred_features is None:
        pred_features = []

    # 전체 요구 컬럼
    required_cols = meta_cols + input_features + lowp_features + output_features + pred_features
    missing_cols = [c for c in required_cols if c not in df.columns]
    if missing_cols:
        raise KeyError(f"❌ 누락된 컬럼 발견: {missing_cols}")

    # 전체 구성 순서 (입력 → 저압입력 → 예측 → 출력)
    df_final = df[meta_cols + input_features + lowp_features + pred_features + output_features].copy()

    # 결측치 제거 (옵션)
    if dropna:
        before = len(df_final)
        df_final.dropna(inplace=True)
        print(f"🧹 결측치 제거: {before - len(df_final)}개 행 삭제 ({len(df_final)}개 남음)")

    # 메타정보 요약
    meta = {
        "n_total": len(df),
        "n_valid": len(df_final),
        "input_features": input_features,
        "lowp_features": lowp_features,
        "pred_features": pred_features,
        "output_features": output_features,
        "meta_columns": meta_cols,
        "dropna": dropna
    }

    # 출력 로그
    print("\n📊 [요약]")
    print(f"입력 피처: {len(input_features)}개")
    print(f"저압 피처: {len(lowp_features)}개")
    print(f"예측 피처: {len(pred_features)}개")
    print(f"출력 피처: {len(output_features)}개")
    print(f"유효 샘플: {len(df_final):,}/{len(df):,}")

    print("\n🔍 [샘플 데이터]")
    print(df_final.head())

    print(f"✅ Dataset 준비 완료 → 입력 {len(input_features)+len(lowp_features)+len(pred_features)}개, 출력 {len(output_features)}개")
    return df_final, meta




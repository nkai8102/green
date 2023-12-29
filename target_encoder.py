import numpy as np
import polars as pl
from sklearn.model_selection import KFold


class TargetEncoder:
    """
    Holdout Target Encoding実行用モジュール
    """
    def __init__(self, kf=None):
        """
        Args:
            kf(sklearn.model_selection._split.KFold): training dataをfold分割する際に使用
        """
        if kf is None:
            kf = KFold(n_splits=5, shuffle=True, random_state=0)
        self.kf = kf
        
        
    def fit_transform(self, cat_df_train, target_series_train):
        """encoderを学習し、さらにKfoldで分割することによるencodeを適用。training dataへの利用を想定。
        Args:
            cat_df_train(polars.dataframe.frame.DataFrame): カテゴリ特徴量のカラムのみを有するDataFrame
            target_series_train(polars.series.series.Series): ターゲット特徴量のカラムを有するSeries
        
        Returns:
            tenc_df_train(polars.dataframe.frame.DataFrame): cat_df_trainにtarget encodingを適用した結果
        """

        # カテゴリ値のため全てstringへ変換
        cat_df_train = cat_df_train.cast(pl.Utf8)

        # target_series_trainはpl.Seriesでなければならない
        assert isinstance(target_series_train, pl.series.series.Series)

        # 1項目ずつ実行
        tenc_df_train = pl.DataFrame({"idx": np.arange(len(cat_df_train))})
        self.mapping_dict = {}
        for col in cat_df_train.columns:
            
            # 学習データ全体で各カテゴリにおけるtargetの平均を計算 (testのencodingで利用される想定)
            self.col_target = target_series_train.name
            xy_train = pl.concat([cat_df_train, target_series_train.to_frame()], how="horizontal")
            target_mean = xy_train.group_by(col).mean()[[col, self.col_target]]
            mapping = dict(target_mean.to_numpy())
            mapping[None] = np.nan # 欠損値はencodingしても欠損値
            self.mapping_dict[col] = mapping 

            # 学習データを分割
            tenc_df_oof_lst = []
            for idx_obj, idx_enc in self.kf.split(xy_train):
                
                # out-of-foldで各カテゴリにおける目的変数の平均を計算
                target_mean = xy_train[idx_obj].group_by(col).mean()[[col, self.col_target]]
                mapping = dict(target_mean.to_numpy())
                mapping[None] = np.nan # 欠損値はencodingしても欠損値

                # Target Encoding (train, out-of-fold)
                # mappingのkeyに含まれていない場合は欠損値とする
                tenc_oof = xy_train[idx_enc, col].cast(pl.Float64).replace(mapping, default=np.nan)
                tenc_oof = tenc_oof.rename(f"{col}_tenc_{self.col_target}")
                tenc_df_oof = pl.DataFrame({"idx": idx_enc}).with_columns(tenc_oof)
                tenc_df_oof_lst.append(tenc_df_oof)

            # 各foldの結果を集約
            tenc_df_train = tenc_df_train.join(pl.concat(tenc_df_oof_lst), on="idx", how="left")

        tenc_df_train = tenc_df_train.drop("idx")
        
        return tenc_df_train


    def transform(self, cat_df_test):
        """学習済のencoderを用いて、入力にencodingを適用。validation/test dataへの利用を想定。
        Args:
            cat_df_test(polars.dataframe.frame.DataFrame): カテゴリ特徴量のカラムのみを有するDataFrame
        
        Returns:
            tenc_df_test(polars.dataframe.frame.DataFrame): cat_df_testにtarget encodingを適用した結果
        """

        # カテゴリ値のため全てstringへ変換
        cat_df_test = cat_df_test.cast(pl.Utf8)

        # 1項目ずつ実行
        tenc_df_test = pl.DataFrame()
        for col in cat_df_test.columns:
            
            # fit_transformモジュール実行時の項目に含まれていない場合、エラーを出力
            assert col in self.mapping_dict.keys()

            # Target Encoding (test)
            # mappingのkeyに含まれていない場合は欠損値とする
            mapping = self.mapping_dict[col]
            tenc_test = cat_df_test[col].cast(pl.Float64).replace(mapping, default=np.nan)
            tenc_test = tenc_test.rename(f"{col}_tenc_{self.col_target}")
            tenc_df_test = tenc_df_test.with_columns(tenc_test)
            
        return tenc_df_test
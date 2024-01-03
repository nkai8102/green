import matplotlib.pyplot as plt
import seaborn as sns


def visualize_distribution(df, x, hue, figsize=(12, 5)):
    """積み上げヒストグラム・カーネル密度分布をクラス別に描画
    Args:
        df(polars.dataframe.frame.DataFrame): 分布データおよびクラス列を有するデータフレーム
        x(str): 分布を描画する対象のカラム名
        hue(str): クラスを表すカラム名
        
    """

    df = df[[x, hue]].to_pandas()

    fig, axes = plt.subplots(1, 2, figsize=figsize)

    # 積み上げヒストグラム
    sns.histplot(data=df, x=x, palette="tab10", hue=hue, 
                ax=axes[0], multiple="stack")
    axes[0].set_xlabel(x)
    axes[0].set_ylabel("Count")

    # カーネル密度分布
    sns.kdeplot(data=df, x=x, palette="tab10", hue=hue, 
                ax=axes[1], common_norm=False)
    axes[1].set_xlabel(x)
    axes[1].set_ylabel("Density")

    plt.show()
    plt.close()
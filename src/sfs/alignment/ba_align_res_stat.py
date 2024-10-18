import pandas as pd
from matplotlib import pyplot as plt

def plot_img_alignments(statfil, outpng=None, threshold_bad_image=0.5):
    df = pd.read_parquet(statfil)
    print(df)
    print(df.columns)
    df = df.astype(
        {'img1': str, 'img2': str, 'mean_dx': float, 'mean_dy': float, 'median_dx': float, 'median_dy': float,
         'std_dx': float, 'std_dy': float, 'nb_ba_matches': 'uint32'})

    # Combine img1 and img2 into a single column for grouping
    combined_imgs = df[['img1', 'median_dx', 'median_dy', 'nb_ba_matches']].rename(columns={"img1": "img"})
    combined_imgs_2 = df[['img2', 'median_dx', 'median_dy', 'nb_ba_matches']].rename(columns={"img2": "img"})

    # Concatenate the two parts
    combined = pd.concat([combined_imgs, combined_imgs_2])

    # Group by the combined image column and compute mean and std for median_dx and median_dy
    df = combined.groupby("img").agg({
        'median_dx': ['mean', 'std'],
        'median_dy': ['mean', 'std'],
        'nb_ba_matches': ['mean']
    }).reset_index()

    df.columns = ['img', 'mean_median_dx', 'std_median_dx', 'mean_median_dy', 'std_median_dy', 'mean_nb_ba_matches']

    bad_imgs = df.loc[(abs(df.mean_median_dx) > threshold_bad_image) | (abs(df.mean_median_dy) > threshold_bad_image)]
    
    # Create figure and axes
    fig, ax = plt.subplots(1, 2, figsize=(20, 10))

    # Plot mean_dx and std_dx on ax[0]
    ax[0].bar(df.index, df['mean_median_dx'], yerr=df['std_median_dx'], capsize=4)
    ax[0].set_title('mean_dx with std_dx')
    ax[0].set_xlabel('NAC Index')
    ax[0].set_ylabel('DX Values')

    # Plot mean_dy and std_dy on ax[1]
    ax[1].bar(df.index, df['mean_median_dy'], yerr=df['std_median_dy'], capsize=4)
    ax[1].set_title('mean_dy with std_dy')
    ax[1].set_xlabel('NAC Index')
    ax[1].set_ylabel('DY Values')

    # Iterate through the rows of the DataFrame and check your condition
    for index, row in df.iterrows():
        if abs(row['mean_median_dx']) > threshold_bad_image/2.:
            ax[0].text(index, row['mean_median_dx'], row['img'], ha='center', va='bottom')
        if abs(row['mean_median_dy']) > threshold_bad_image/2.:
            ax[1].text(index, row['mean_median_dy'], row['img'], ha='center', va='bottom')

    plt.tight_layout()

    if outpng == None:
        plt.show()
    else:
        plt.savefig(outpng)

    return bad_imgs, df

if __name__ == '__main__':

    pd.set_option('display.max_columns', None)
    statfil = "/home/sberton2/Scaricati/stats_DM2b4_0_ba2.parquet"

    bad_images = plot_img_alignments(statfil)
    print(bad_images)

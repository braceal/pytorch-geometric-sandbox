import time
from plotly.io import to_html
from molecules.plot.tsne import compute_tsne, plot_tsne_plotly


def tsne_validation(embeddings, paint, paint_name, plot_dir, plot_name):
    print(f"t-SNE on input shape {embeddings.shape}")
    tsne_embeddings = compute_tsne(embeddings)
    fig = plot_tsne_plotly(
        tsne_embeddings, df_dict={paint_name: paint}, color=paint_name
    )
    html_string = to_html(fig)
    time_stamp = time.strftime(f"{plot_name}-%Y%m%d-%H%M%S.html")
    with open(plot_dir.joinpath(time_stamp), "w") as f:
        f.write(html_string)

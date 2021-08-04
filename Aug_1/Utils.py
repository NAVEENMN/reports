import os
import matplotlib.pyplot as plt

class Utils(object):

    @classmethod
    def create_gif(cls, loc='pc_simulation'):
        import os
        import glob
        from PIL import Image

        fcont = len(glob.glob(f"{os.getcwd()}/tmp/graph_*.png"))
        # ref: https://pillow.readthedocs.io/en/stable/handbook/image-file-formats.html#gif
        img, *imgs = [Image.open(f"{os.getcwd()}/tmp/graph_{i}.png") for i in range(1, fcont)]
        img.save(fp=f"{os.getcwd()}/media/{loc}.gif",
                 format='GIF',
                 append_images=imgs,
                 save_all=True,
                 duration=10,
                 loop=0)

        # delete all png files.
        fp_in = f"{os.getcwd()}/tmp/graph_*.png"
        for f in glob.glob(fp_in):
            os.remove(f)

    @classmethod
    def save_pair_plot(cls, observations):
        import os
        import seaborn as sns
        sns.pairplot(observations).savefig(f"{os.getcwd()}/media/relationship.png")

    @classmethod
    def save_graph(cls, causal_graph, testing_graph, predicted_graph, step, attr=None):
        fig, axes = plt.subplots(1, 3, figsize=(16, 8))
        axes[0].set_title('Original Graph')
        causal_graph.draw(axes=axes[0])
        axes[1].set_title(f'{attr[0]} = {attr[1]}')
        Graph.draw_graph(testing_graph, axes=axes[1])
        axes[2].set_title('Predicted Graph')
        Graph.draw_graph(predicted_graph, axes=axes[2])
        # plt.show()
        fig.savefig(os.path.join(os.getcwd(), 'tmp', f'graph_{step}.png'))
        plt.clf()
        plt.close(fig)


ut = Utils()
ut.create_gif()
print('Done.')

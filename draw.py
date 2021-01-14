import os
import click
import foodwebs as fw


@click.command()
@click.option('--scor_dir', help='Directory with SCOR files.')
@click.option('--output', help='Directory where plots should be saved.')
@click.option('--boundary', default=False, is_flag=True, help='Wheter to show boundary flows.')
@click.option('--show_trophic_layer', default=True, is_flag=True, help='Wheter to show trophic layer.')
@click.option('--switch_axes', default=False, is_flag=True, help='Wheter to switch axes.')
@click.option('--normalization', default=None, type=click.Choice(['log', 'diet', 'biomass', 'tst']),
              help='Normalization method.')
def draw_heatmaps(scor_dir, output, boundary, show_trophic_layer, switch_axes, normalization):
    for subdir, dirs, files in os.walk(scor_dir):
        for f in files:
            print(f'Processing: {f}...')
            food_web = fw.read_from_SCOR(os.path.join(scor_dir, f))

            fig = fw.draw_heatmap(food_web,
                                  boundary=boundary,
                                  normalization=normalization,
                                  show_trophic_layer=show_trophic_layer,
                                  switch_axes=switch_axes,
                                  show_plot=False)
            fig.write_image(f'{output}/{f}_heatmap.png')

            fig = fw.draw_trophic_flows_distribution(food_web, show_plot=False)
            fig.write_image(f'{output}/{f}_distribution.png')


if __name__ == "__main__":
    draw_heatmaps()

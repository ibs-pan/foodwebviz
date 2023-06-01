'''Simple script that generates plots for all foodwebs from given directory (containing SCOR files).

For each foodweb in directory the following files will created:
    - heatmap
    - trophic flows distribution
    - network visualisation (only if foodweb has less than 20 nodes)


Example usage:
python3.8 draw.py --scor_dir data/ --output plots/
'''

import os
import click
import foodwebviz as fw


@click.command('Generates plots for all foodwebs from given directory (containing SCOR files).')
@click.option('--scor_dir', help='Directory with SCOR files.')
@click.option('--output', help='Directory where plots should be saved.', default='.')
@click.option('--boundary', default=False, is_flag=True, help='Wheter to show boundary flows.')
@click.option('--show_trophic_layer', default=True, is_flag=True, help='Wheter to show trophic layer.')
@click.option('--switch_axes', default=False, is_flag=True, help='Wheter to switch axes.')
@click.option('--normalization', default=None, type=click.Choice(['log', 'diet', 'biomass', 'tst']),
              help='Normalization method.')
def draw_heatmaps(scor_dir, output, boundary, show_trophic_layer, switch_axes, normalization):
    '''Generates plots for all foodwebs from given directory (containing SCOR files).

    For each foodweb in directory the following files will created:

        * heatmap

        * trophic flows distribution

        * network visualization (only if foodweb has less than 20 nodes)
    '''
    for subdir, dirs, files in os.walk(scor_dir):
        for f in files:
            print(f'Processing: {f}...')
            food_web = fw.read_from_SCOR(os.path.join(scor_dir, f))

            fig = fw.draw_heatmap(food_web,
                                  boundary=boundary,
                                  normalization=normalization,
                                  show_trophic_layer=show_trophic_layer,
                                  switch_axes=switch_axes,
                                  height=1000)
            fig.write_image(f'{output}/{f}_heatmap.png')

            fig = fw.draw_trophic_flows_distribution(food_web)
            fig.write_image(f'{output}/{f}_throphic_levels_distribution.png')

            fig = fw.draw_trophic_flows_heatmap(food_web)
            fig.write_image(f'{output}/{f}_trophic_levels_heatmap.png')

            if food_web.n <= 20:
                fw.draw_network_for_nodes(food_web,
                                          file_name=f'{output}/{f}_network.html',
                                          notebook=False)


if __name__ == "__main__":
    draw_heatmaps()

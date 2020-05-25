
import click

from masked_transformer.commands import train, evaluate


@click.group(invoke_without_command=True)
def main(**kwargs):
    pass


main.add_command(train.main, 'train')
main.add_command(evaluate.main, 'evaluate')

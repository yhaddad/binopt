# -*- coding: utf-8 -*-
"""Clic interface."""

import click
from binopt import core as cbin


@click.command()
@click.option('--shape', default=1, help='Number of greetings.')
@click.option('--test',  prompt='Are you testing', default='y',
              help='The person to greet.')
def optimise(count, test):
    """Simple program that greets NAME for a total of COUNT times."""
    if test is 'y':
        click.echo("you have these modules : %s" % ''.join(dir(cbin)))


if __name__ == '__main__':
    optimise()

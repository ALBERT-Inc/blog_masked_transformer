import click
from ruamel import yaml
from torch.utils.data import DataLoader
from ignite.contrib.handlers import ProgressBar

from masked_transformer.dataset import YouCook
from masked_transformer.model import (TemporalCoder, MaskedTransformer,
                                      MaskedTransformerLoss)
from masked_transformer.commands._trainer import (create_densecap_evaluator)
from masked_transformer.metrics import DenseCaptionBleu


def get_model(vocab, net_params=None, loss_params=None, coder_params=None,
              **kwargs):
    net_params = net_params if net_params else {}
    loss_params = loss_params if loss_params else {}

    coder = TemporalCoder(**coder_params)
    net = MaskedTransformer(coder, vocab, **net_params)
    loss = MaskedTransformerLoss(coder=coder, vocab=vocab, **loss_params)
    return net, loss


@click.command()
@click.argument('config_file', type=click.Path(exists=True))
@click.option('--dataset_root', type=click.Path(exists=True), default="data")
@click.option('--device', default=None)
@click.option('--num_workers', type=int, default=0)
def main(config_file, dataset_root, device, num_workers):
    with open(config_file) as stream:
        config = yaml.safe_load(stream)

    text_proc = YouCook.get_text_proc(dataset_root)
    vocab = text_proc.vocab
    val_dataset = YouCook(dataset_root, text_proc, image_set='val')
    val_loader = DataLoader(val_dataset,
                            batch_size=1,
                            num_workers=num_workers,
                            collate_fn=val_dataset.collate_fn,
                            shuffle=False)
    model, _ = get_model(vocab, **config['model'])

    metrics = {'bleu': DenseCaptionBleu()}
    evaluator = create_densecap_evaluator(model, metrics, device)
    ProgressBar(persist=True).attach(evaluator)

    evaluator.run(val_loader)
    print(evaluator.state.metrics)


if __name__ == "__main__":
    main()

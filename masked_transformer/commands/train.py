import click
from ruamel import yaml
import torch
from torch.utils.data import DataLoader

from masked_transformer.dataset import YouCook
from masked_transformer.model import (TemporalCoder, MaskedTransformer,
                                      MaskedTransformerLoss)
from masked_transformer.commands._trainer import (create_densecap_trainer,
                                                  TrainExtension)


def get_model(vocab, net_params, loss_params, coder_params, **kwargs):
    net_params = net_params if net_params else {}
    loss_params = loss_params if loss_params else {}

    coder = TemporalCoder(**coder_params)
    net = MaskedTransformer(coder, vocab, **net_params)
    loss = MaskedTransformerLoss(coder=coder, vocab=vocab, **loss_params)
    return net, loss


@click.command()
@click.argument('config_file', type=click.Path(exists=True))
@click.option('--dataset_root', type=click.Path(exists=True), default="data")
@click.option('--res_dir', type=click.Path(exists=False),
              default='results')
@click.option('--device', default=None)
@click.option('--num_workers', type=int, default=0)
def main(config_file, dataset_root, res_dir, device, num_workers):
    with open(config_file) as stream:
        config = yaml.safe_load(stream)

    text_proc = YouCook.get_text_proc(dataset_root)
    vocab = text_proc.vocab
    train_dataset = YouCook(dataset_root, text_proc, image_set='train')
    train_loader = DataLoader(train_dataset,
                              config['batchsize'],
                              num_workers=num_workers,
                              collate_fn=train_dataset.collate_fn,
                              shuffle=True)
    model, criterion = get_model(vocab, **config['model'])
    optimizer = torch.optim.Adam(model.parameters(), **config['optimizer'])

    trainer = create_densecap_trainer(model, optimizer, criterion, device)
    train_extend = TrainExtension(trainer, res_dir)
    train_extend.print_metrics()
    train_extend.set_progressbar()
    train_extend.save_model(model, **config['model_checkpoint'])
    trainer.run(train_loader, max_epochs=config['epochs'])


if __name__ == "__main__":
    main()

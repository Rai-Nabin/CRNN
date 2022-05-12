import argparse

import torch
from tqdm import tqdm
from torch.utils.data import DataLoader

from src.config import common_config as config
from src.ctc_decoder import ctc_decode
from src.dataset import Synth90kDataset
from src.model import CRNN


def predict(crnn, dataloader, label2char, decode_method, beam_size):

    # Set model to evaluation  mode.
    crnn.eval()

    progress_bar = tqdm(total=len(dataloader), desc="Predict")
    
    with torch.no_grad():
        for data in dataloader:
            breakpoint()
            device = 'cuda' if next(crnn.parameters()).is_cuda else 'cpu'
            
            images = data.to(device)
            logits = crnn(images)
            log_probs = torch.nn.Functional.log_softmax(logits, dim=2)
            prediction = ctc_decode(
                log_probs, method=decode_method, beam_size=beam_size, label2char=label2char)
            all_predictions += prediction

            progress_bar.update(1)
        progress_bar.close()
    return all_predictions


def show_result(paths, predictions):
    print("******Result*******")
    for path, prediction in zip(paths, predictions):
        text = ''.join(prediction)
        print(f'{path} > {text}')


def main(image_path, checkpoints, batch_size, decode, beam_size):

    # Set the computation device.
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Available Device: {device}')

    # Load image dataset to predict.
    predict_dataset = Synth90kDataset(
        paths=image_path, image_height=config['image_height'], image_width=config['image_width'])
   
    predict_loader = DataLoader(
        dataset=predict_dataset, batch_size=batch_size, shuffle=False)
    
    num_classes = len(Synth90kDataset.LABEL2CHAR) + 1

    # Construct a CRNN model.
    crnn = CRNN(image_channel=1, image_height=config['image_height'], image_width=config['image_width'], num_classes=num_classes,
                map_to_seq_hidden=config['map_to_seq_hidden'], rnn_hidden=config['rnn_hidden'], leaky_relu=config['leaky_relu'])
   
    # Load the model on to the computation device.
    crnn.to(device)

    # Load checkpoints on to the computation device.
    crnn.load_state_dict(torch.load(checkpoints, map_location=device))
    
    predictions = predict(crnn, predict_loader, num_classes-1,
                          decode_method=decode, beam_size=beam_size)
    breakpoint()
    show_result(image_path, predictions)


if __name__ == "__main__":
    # Create the parser.
    parser = argparse.ArgumentParser()

    # Add arguments.
    parser.add_argument('--image', required=True,
                        help='Path to the input image.')
    parser.add_argument('--checkpoints', default='checkpoints/synth90k.pt',
                        help='Path to the checkpoints.')
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--decode', default='beam_search',
                        help="Decoding method.")
    parser.add_argument('--beam_size', default=10, type=int)

    # Parse the arguments.
    args = vars(parser.parse_args())

    image_path = args['image']
    checkpoints = args['checkpoints']
    batch_size = args['batch_size']
    decode = args['decode']
    beam_size = args['beam_size']

    main(image_path, checkpoints, batch_size, decode, beam_size)

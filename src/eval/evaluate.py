import os
import pandas as pd
import torch

from tqdm import tqdm
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader

TOTAL_NUMBER_OF_SUBMISSIONS = 6
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


class AverageMeter(object):

    def __init__(self):
        self.reset()

    def reset(self):
        self.avg = 0
        self.sum = 0
        self.cnt = 0

    def update(self, val, n=1):
        self.sum += val * n
        self.cnt += n
        self.avg = self.sum / self.cnt


def accuracy(logits, labels):
    preds = torch.argmax(logits, dim=1)
    return torch.true_divide(torch.sum(preds == labels), len(labels))


def eval_fn(model, loader, device):
    """
    Evaluation method
    :param model: model to evaluate
    :param loader: data loader for either training or testing set
    :param device: torch device
    :return: accuracy on the data
    """
    score = AverageMeter()
    model.eval()

    t = tqdm(loader)
    confusion_matrix = torch.zeros(len(loader.dataset.classes), len(loader.dataset.classes))
    with torch.no_grad():  # no gradient needed
        for images, labels in t:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            for t, p in zip(labels.view(-1), outputs.argmax(dim=1).view(-1)):
                confusion_matrix[(t-1).long(), (p-1).long()] += 1
            confusion_matrix = confusion_matrix / confusion_matrix.sum(1)
            #save confusion matrix in a file
            torch.save(confusion_matrix, 'confusion_matrix.pt')
            acc = accuracy(outputs, labels)
            score.update(acc.item(), images.size(0))

            t.set_description('(=> Test) Score: {:.4f}'.format(score.avg))

    return score.avg

def generate_latex_table(df, track):
    latex_template = r"""
\begin{{table}}
\centering
\caption{{{track} track leaderboard}}
\label{{tab:results}}
\begin{{tabular}}{{|c|c|c|c|}}
\hline
\textbf{{Rank}} & \textbf{{Team}} & \textbf{{Score}} & \textbf{{Nr. of submissions left}} \\
\hline
{table_rows}
\hline
\end{{tabular}}
\end{{table}}
"""
    table_rows = ""
    for _, row in df.iterrows():
        table_rows += f"{row['Rank']} & {row['Team']} & {row['Score']} & {row['Nr. submissions left']} \\\\ \n"

    latex_code = latex_template.format(track=track, table_rows=table_rows)

    with open(f'{track}_track_results.tex', 'w') as f:
        f.write(latex_code)
    
    print('Latex table saved to', f'{track}_track_results.tex')

def save_results(track, team_name, score) -> pd.DataFrame:

    df = pd.read_csv(f'{track}_track_results.csv') if os.path.exists(f'{track}_track_results.csv') else pd.DataFrame()

    if df.empty:
        df = pd.DataFrame(columns=['Team', 'Score', 'Nr. submissions left'])

    if team_name in df['Team'].values:
        df.loc[df['Team'] == team_name, 'Score'] = score
        df.loc[df['Team'] == team_name, 'Nr. submissions left'] -= 1
    else:
        new_row = pd.DataFrame({'Team': [team_name], 'Score': [score], 'Nr. submissions left': [TOTAL_NUMBER_OF_SUBMISSIONS - 1]})
        df = pd.concat([df, new_row], ignore_index=True)

    df = df.sort_values(by='Score', ascending=False).reset_index(drop=True)

    df['Rank'] = df['Score'].rank(method='min', ascending=False).astype(int)

    df.to_csv(f'{track}_track_results.csv', index=False)
    print('Results saved to', f'{track}_track_results.csv')

    return df

def eval_model(model, saved_model_file, test_data_dir, data_augmentations) -> float:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model.load_state_dict(torch.load(os.path.join(os.getcwd(), 'models', saved_model_file), map_location=device))
    print("Number of parameters: ", sum(p.numel() for p in model.parameters()))
    data = ImageFolder(test_data_dir, transform=data_augmentations)

    test_loader = DataLoader(dataset=data,
                             batch_size=128,
                             shuffle=False)

    score = eval_fn(model, test_loader, device)

    print('Avg accuracy:', str(score*100) + '%')
    return score
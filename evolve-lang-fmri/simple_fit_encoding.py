"""Simple fMRI encoding pipeline: predict Huth-dataset fMRI responses to language
from GPT-2 XL final-token 10-gram features.

A pared-down version of `~/automated-brain-explanations/experiments/02_fit_encoding.py`
that only depends on the local `src/` package.

Run a pilot:
    uv run simple_fit_encoding.py --num_train 2 --num_test 1
"""
import argparse
import os
import time
from os.path import join

import joblib
import numpy as np

from src import data, encoding, features


def main(args):
    t0 = time.time()
    train_stories, test_stories = data.get_story_names(
        num_train=args.num_train, num_test=args.num_test)
    print(f'train stories ({len(train_stories)}):', train_stories)
    print(f'test stories ({len(test_stories)}):', test_stories)
    all_stories = train_stories + test_stories

    print('loading wordseqs + responses...')
    wordseqs = data.load_wordseqs(all_stories)
    resps = data.load_responses(all_stories, subject=args.subject)

    print('extracting GPT-2 XL features...')
    embedder = features.GPT2Embedder(
        checkpoint=args.checkpoint, layer=args.layer, device=args.device)
    stim_train = features.get_features(
        wordseqs, train_stories, embedder, ngram_size=args.ngram_size, ndelays=args.ndelays)
    stim_test = features.get_features(
        wordseqs, test_stories, embedder, ngram_size=args.ngram_size, ndelays=args.ndelays)

    resp_train = np.vstack([resps[s] for s in train_stories])
    resp_test = np.vstack([resps[s] for s in test_stories])
    print('feature shapes', stim_train.shape, stim_test.shape)
    print('response shapes', resp_train.shape, resp_test.shape)
    assert stim_train.shape[0] == resp_train.shape[0], 'train feat/resp TR mismatch'
    assert stim_test.shape[0] == resp_test.shape[0], 'test feat/resp TR mismatch'

    print('fitting ridge encoding model...')
    r = encoding.fit_encoding(
        stim_train, resp_train, stim_test, resp_test,
        nboots=args.nboots, chunklen=args.chunklen, nchunks=args.nchunks)

    r.update({k: v for k, v in vars(args).items()})
    r['train_stories'] = train_stories
    r['test_stories'] = test_stories
    print(f"\ntest corr  mean={r['corrs_test_mean']:.4f}  "
          f"median={r['corrs_test_median']:.4f}  "
          f"frac>0={r['corrs_test_frac>0']:.4f}  "
          f"top5%={r['corrs_test_mean_top5_percentile']:.4f}")

    os.makedirs(args.save_dir, exist_ok=True)
    out = join(args.save_dir, f'results_{args.subject}_layer{args.layer}.pkl')
    joblib.dump(r, out)
    print(f'saved to {out}  ({(time.time() - t0) / 60:.1f} min)')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--subject', type=str, default='UTS03')
    parser.add_argument('--num_train', type=int, default=2,
                        help='number of training stories (pilot uses 2)')
    parser.add_argument('--num_test', type=int, default=1,
                        help='number of test stories (pilot uses 1)')
    parser.add_argument('--checkpoint', type=str, default='gpt2-xl')
    parser.add_argument('--layer', type=int, default=24,
                        help='GPT-2 XL hidden-state layer to use (0..48)')
    parser.add_argument('--ngram_size', type=int, default=10)
    parser.add_argument('--ndelays', type=int, default=4)
    parser.add_argument('--nboots', type=int, default=5)
    parser.add_argument('--chunklen', type=int, default=40)
    parser.add_argument('--nchunks', type=int, default=20)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--save_dir', type=str,
                        default=join(os.path.dirname(os.path.abspath(__file__)), 'results'))
    main(parser.parse_args())

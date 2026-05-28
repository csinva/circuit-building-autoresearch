"""Load the Huth fMRI language dataset (wordseqs + fMRI responses).

Simplified from `neuro/data/{stim_utils,response_utils,story_names}.py`.
Uses the pre-packaged "huge_data" joblib files, which are the fastest to load
and are already aligned: each story's response is trimmed by [10:-5] relative
to its TR timeline (so downsampled features must be trimmed identically).
"""
import os
import sys
from os.path import join

import joblib
import numpy as np

# the cached wordseqs were pickled referencing the top-level `ridge_utils`
# package, so make our shim importable before any joblib.load call
sys.path.insert(0, os.path.dirname(__file__))
import ridge_utils  # noqa: E402,F401

FMRI_DIR = '/home/chansingh/mntv1/deep-fMRI/data'
HUGE_DIR = join(FMRI_DIR, 'huge_data')

# the training stories shared across the UTS01/UTS02/UTS03 "huge" training sets
# (intersection of all three), plus the standard held-out test stories. Using the
# shared set keeps the same stimuli usable for any of the three subjects.
TRAIN_STORIES = [
    'itsabox', 'odetostepfather', 'inamoment', 'hangtime', 'ifthishaircouldtalk',
    'goingthelibertyway', 'golfclubbing', 'thetriangleshirtwaistconnection',
    'igrewupinthewestborobaptistchurch', 'tetris', 'becomingindian',
    'canplanetearthfeedtenbillionpeoplepart1', 'thetiniestbouquet', 'swimmingwithastronauts',
    'lifereimagined', 'forgettingfear', 'stumblinginthedark', 'backsideofthestorm', 'food',
    'theclosetthatateeverything', 'notontheusualtour', 'exorcism', 'adventuresinsayingyes',
    'thefreedomridersandme', 'cocoonoflove', 'waitingtogo', 'thepostmanalwayscalls',
    'googlingstrangersandkentuckybluegrass', 'mayorofthefreaks', 'learninghumanityfromdogs',
    'shoppinginchina', 'souls', 'cautioneating', 'comingofageondeathrow',
    'breakingupintheageofgoogle', 'gpsformylostidentity', 'eyespy', 'treasureisland',
    'thesurprisingthingilearnedsailingsoloaroundtheworld', 'theadvancedbeginner',
    'goldiethegoldfish', 'life', 'thumbsup', 'seedpotatoesofleningrad', 'theshower', 'adollshouse',
    'canplanetearthfeedtenbillionpeoplepart2', 'sloth', 'howtodraw', 'quietfire', 'metsmagic',
    'penpal', 'thecurse', 'canadageeseandddp', 'thatthingonmyarm', 'buck',
    'wildwomenanddancingqueens', 'againstthewind', 'indianapolis', 'alternateithicatom', 'bluehope',
    'kiksuya', 'afatherscover', 'haveyoumethimyet', 'firetestforlove',
    'catfishingstrangerstofindmyself', 'christmas1940', 'tildeath', 'lifeanddeathontheoregontrail',
    'vixenandtheussr', 'undertheinfluence', 'beneaththemushroomcloud', 'jugglingandjesus',
    'superheroesjustforeachother', 'sweetaspie', 'naked', 'singlewomanseekingmanwich', 'avatar',
    'whenmothersbullyback', 'myfathershands', 'reachingoutbetweenthebars', 'theinterview',
    'stagefright', 'legacy', 'canplanetearthfeedtenbillionpeoplepart3', 'listo',
    'gangstersandcookies', 'birthofanation', 'mybackseatviewofagreatromance',
    'lawsthatchokecreativity', 'threemonths', 'whyimustspeakoutaboutclimatechange',
    'leavingbaghdad',
]
TEST_STORIES = ['fromboyhoodtofatherhood', 'wheretheressmoke', 'onapproachtopluto']


def get_story_names(num_train=None, num_test=1):
    """Return (train_stories, test_stories)."""
    train = TRAIN_STORIES if num_train is None else TRAIN_STORIES[:num_train]
    return train, TEST_STORIES[:num_test]


def load_wordseqs(stories):
    """Return {story: DataSequence} with .data (words), .data_times, .tr_times."""
    wordseqs = joblib.load(join(HUGE_DIR, 'wordseqs.joblib'))
    return {s: wordseqs[s] for s in stories}


def load_responses(stories, subject='UTS03'):
    """Return {story: (n_trs, n_voxels)} fMRI responses (already trimmed [10:-5])."""
    resps = joblib.load(join(HUGE_DIR, f'{subject}_responses.jbl'))
    return {s: resps[s] for s in stories}

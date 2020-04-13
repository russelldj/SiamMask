# --------------------------------------------------------
# SiamMask
# Licensed under The MIT License
# Written by Qiang Wang (wangqiang2015 at ia.ac.cn)
# --------------------------------------------------------
from tools.test import *
from custom import Custom

import glob
import pdb
from argparse import Namespace

#parser = argparse.ArgumentParser(description='PyTorch Tracking Demo')
#
#parser.add_argument('--resume', default='', type=str, required=True,
#                    metavar='PATH',help='path to latest checkpoint (default: none)')
#parser.add_argument('--config', dest='config', default='config_davis.json',
#                    help='hyper-parameter of SiamMask in json format')
#parser.add_argument('--base_path', default='../../data/tennis', help='datasets')
#parser.add_argument('--cpu', action='store_true', help='cpu mode')
#args = parser.parse_args()
#pdb.set_trace()

DEFAULT_RESUME = "HATracking/libs/SiamMask/experiments/siammask_sharp/SiamMask_DAVIS.pth"
DEFAULT_CONFIG = "HATracking/libs/SiamMask/experiments/siammask_sharp/config_davis.json"
DEFAULT_BASE_PATH = "HATracking/libs/SiamMask/data/tennis"


class SiamMaskWrapper():
    def __init__(self, base_path=DEFAULT_BASE_PATH, config=DEFAULT_CONFIG,
                 resume=DEFAULT_RESUME, cpu=False):
        args = Namespace(base_path=base_path, config=config,
                         resume=resume, cpu=cpu)

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        torch.backends.cudnn.benchmark = True
        self.state = None
        self.cfg = load_config(args)  # TODO figure out the important parts of this
        self.siammask = Custom(anchors=self.cfg['anchors'])
        if args.resume:
            assert isfile(args.resume), 'Please download {} first.'.format(args.resume)
            self.siammask = load_pretrain(self.siammask, args.resume)

        self.siammask.eval().to(self.device)

    def select_region(self, image, xywh=None):
        """
        image : 3 channel image
            The initial image with the object
        xywh : ArrayLike
            the position of the initial bounding rectangle as [x, y, w, h]
            If unspecified, a pop up selection will be used
        """
        if xywh is None:
            xywh = cv2.selectROI('SiamMask', image, False, False)

        x, y, w, h = xywh  # simply expand for convenience

        target_pos = np.array([x + w / 2, y + h / 2])
        target_sz = np.array([w, h])
        # init tracker
        self.state = siamese_init(image, target_pos, target_sz, self.siammask,
                                  self.cfg['hp'], device=self.device)

    def predict(self, image, visualize=False, verbose=False):
        self.state = siamese_track(self.state, image, mask_enable=True,
                                   refine_enable=True, device=self.device)

        target_pos = self.state["target_pos"]
        target_sz = self.state["target_sz"]
        score = self.state["score"]
        location = self.state['ploygon'].flatten()
        # compute as ltwh
        ltwh = np.concatenate((location[0:2], location[4:6] - location[0:2]))
        transformed_loc = [np.int0(location).reshape((-1, 1, 2))]
        if verbose:
            print("transformed loc : {}".format(transformed_loc))

        mask = self.state['mask'] > self.state['p'].seg_thr
        image[:, :, 2] = (mask > 0) * 255 + (mask == 0) * image[:, :, 2]
        cv2.polylines(image, transformed_loc, True, (0, 255, 0), 3)
        #cv2.line(image, tuple(loc[0:2]), tuple(loc[0:2] + loc[2:]), (0, 255, 0))
        if visualize:
            # return mask
            cv2.imshow('SiamMask', image)
            cv2.waitKey(10000)

        return ltwh, score, image  # TODO the image should be a crop

#if __name__ == '__main__':
#    # Setup device
#    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#    torch.backends.cudnn.benchmark = True
#
#    # Setup Model
#    cfg = load_config(args)
#    from custom import Custom
#    siammask = Custom(anchors=cfg['anchors'])
#    if args.resume:
#        assert isfile(args.resume), 'Please download {} first.'.format(args.resume)
#        siammask = load_pretrain(siammask, args.resume)
#
#    siammask.eval().to(device)
#
#    # Parse Image file
#    img_files = sorted(glob.glob(join(args.base_path, '*.jp*')))
#    ims = [cv2.imread(imf) for imf in img_files]
#
#    # Select ROI
#    cv2.namedWindow("SiamMask", cv2.WND_PROP_FULLSCREEN)
#    # cv2.setWindowProperty("SiamMask", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
#    try:
#        init_rect = cv2.selectROI('SiamMask', ims[0], False, False)
#        x, y, w, h = init_rect
#    except:
#        exit()
#
#    toc = 0
#    for f, im in enumerate(ims):
#        tic = cv2.getTickCount()
#        if f == 0:  # init
#            target_pos = np.array([x + w / 2, y + h / 2])
#            target_sz = np.array([w, h])
#            state = siamese_init(im, target_pos, target_sz, siammask, cfg['hp'], device=device)  # init tracker
#        elif f > 0:  # tracking
#            state = siamese_track(state, im, mask_enable=True, refine_enable=True, device=device)  # track
#            location = state['ploygon'].flatten()
#            mask = state['mask'] > state['p'].seg_thr
#
#            im[:, :, 2] = (mask > 0) * 255 + (mask == 0) * im[:, :, 2]
#            cv2.polylines(im, [np.int0(location).reshape((-1, 1, 2))], True, (0, 255, 0), 3)
#            cv2.imshow('SiamMask', im)
#            key = cv2.waitKey(1)
#            if key > 0:
#                break
#
#        toc += cv2.getTickCount() - tic
#    toc /= cv2.getTickFrequency()
#    fps = f / toc
#    print('SiamMask Time: {:02.1f}s Speed: {:3.1f}fps (with visulization!)'.format(toc, fps))

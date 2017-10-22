from collections import deque
import numpy as np
from sklearn.metrics import mean_squared_error


class Lane:
    def __init__(self, thresh=500000):
        # was the line detected in the last iteration?
        self.detected = False
        self.flags = {'curv': False, 'off_center': False, 'left_fit': False, 'right_fit': False}

        self._left_fit_buf = deque(maxlen=10)
        self._right_fit_buf = deque(maxlen=10)

        self.left_fit = []
        self.right_fit = []

        self.curvature = None
        self.off_center = None

        self._curv_buf = deque(maxlen=10)
        self._off_center_buf = deque(maxlen=10)

        self.thresh = thresh

    def get_mean_points(self, side='left'):
        if side is 'left':
            return np.mean(np.array(list(self._left_fit_buf)), axis=0)

        if side is 'right':
            return np.mean(np.array(list(self._right_fit_buf)), axis=0)

    def _update_curvature(self, curvature):
        # update curvature
        if len(self._curv_buf) < self._curv_buf.maxlen:
            self._curv_buf.append(curvature)
            print('filling curvature buf', len(self._curv_buf), '/', self._curv_buf.maxlen)
        else:
            # sanity check
            data = np.array(list(self._curv_buf))
            self.curvature = np.mean(data)
            std = np.std(data)
            new_std = np.std(np.append(data, curvature))
            print('cur: std=', std, 'new_std=', new_std)

            if (new_std > 3 * std):
                print('curv: reject, std_increase = ', (new_std - std) / std)
                self.flags['curv'] = False
            else:
                print('curv: update')
                self._curv_buf.append(curvature)
                self.flags['curv'] = True

    def _update_off_center(self, off_center):
        if len(self._off_center_buf) < self._off_center_buf.maxlen:
            self._off_center_buf.append(off_center)
            print('filling off_center buf', len(self._off_center_buf), '/', self._off_center_buf.maxlen)
        else:
            # sanity check
            data = np.array(list(self._off_center_buf))
            self.off_center = np.mean(data)
            std = np.std(data)
            new_std = np.std(np.append(data, off_center))
            print('off: std=', std, 'new_std=', new_std)

            if (new_std > 6 * std):
                print('off_center: reject, std_increase = ', (new_std - std) / std)
                self.flags['off_center'] = False
            else:
                print('off_center: update')
                self._off_center_buf.append(off_center)
                self.flags['off_center'] = True

    def _update_fit(self, new_fit, side='left'):
        # update fit

        if side is 'left':
            if len(self._left_fit_buf) < self._left_fit_buf.maxlen:
                self._left_fit_buf.append(new_fit)
                print('filing left_fit buff', len(self._left_fit_buf), '/', self._left_fit_buf.maxlen)
            else:
                # sanity check
                self.left_fit = self.get_mean_points('left')
                mse = mean_squared_error(new_fit, self.left_fit)

                if mse > self.thresh:
                    print('left_fit: reject, MSE = ', mse)
                    self.flags['left_fit'] = False
                else:
                    print('left_fit: update')
                    self._left_fit_buf.append(new_fit)
                    self.flags['left_fit'] = True

        if side is 'right':
            if len(self._right_fit_buf) < self._right_fit_buf.maxlen:
                self._right_fit_buf.append(new_fit)
                print('filing right_fit buff', len(self._right_fit_buf), '/', self._right_fit_buf.maxlen)
            else:
                # sanity check
                self.right_fit = self.get_mean_points('right')
                mse = mean_squared_error(new_fit, self.right_fit)

                if mse > self.thresh:
                    print('right_fit: reject, MSE = ', mse)
                    self.flags['right_fit'] = False
                else:
                    print('right_fit: update')
                    self._right_fit_buf.append(new_fit)
                    self.flags['right_fit'] = True

    def update(self, left_fit, right_fit, curvature, off_center):
        self._update_curvature(curvature)
        self._update_off_center(off_center)
        self._update_fit(left_fit, side='left')
        self._update_fit(right_fit, side='right')

        # update detected state
        if self.flags['left_fit'] and self.flags['right_fit'] is True:
            self.detected = True
        else:
            self.detected = False

        print('Flags: ', self.flags)

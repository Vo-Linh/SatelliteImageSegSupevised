# Obtained from: https://github.com/open-mmlab/mmsegmentation/tree/v0.16.0

import os.path as osp

import torch.distributed as dist
from mmcv.runner import DistEvalHook as _DistEvalHook
from mmcv.runner import EvalHook as _EvalHook
from torch.nn.modules.batchnorm import _BatchNorm


class EvalHook(_EvalHook):
    """Single GPU EvalHook, with efficient test support.

    Args:
        by_epoch (bool): Determine perform evaluation by epoch or by iteration.
            If set to True, it will perform by epoch. Otherwise, by iteration.
            Default: False.
        efficient_test (bool): Whether save the results as local numpy files to
            save CPU memory during evaluation. Default: False.
    Returns:
        list: The prediction results.
    """

    greater_keys = ['mIoU', 'mAcc', 'aAcc']

    def __init__(self, *args, by_epoch=False, efficient_test=False, **kwargs):
        super().__init__(*args, by_epoch=by_epoch, **kwargs)
        self.efficient_test = efficient_test

    def _do_evaluate(self, runner):
        """perform evaluation and save ckpt."""
        if not self._should_evaluate(runner):
            return

        from mmseg.apis import single_gpu_test
        results = single_gpu_test(
            runner.model,
            self.dataloader,
            show=False,
            efficient_test=self.efficient_test)
        runner.log_buffer.output['eval_iter_num'] = len(self.dataloader)
        key_score = self.evaluate(runner, results)
        if self.save_best:
            self._save_ckpt(runner, key_score)

    def _save_ckpt(self, runner, key_score):
        """Save checkpoint with best score.

        This override is needed because IterBasedRunner.save_checkpoint()
        doesn't accept 'save_best' parameter in newer MMCV versions.
        """
        if self.by_epoch:
            current = f'epoch_{runner.epoch + 1}'
            cur_type, cur_time = 'epoch', runner.epoch + 1
        else:
            current = f'iter_{runner.iter + 1}'
            cur_type, cur_time = 'iter', runner.iter + 1

        best_score = runner.meta.get('hook_msgs', {}).get(
            'best_score', self.key_indicator == 'auto' and 0 or -float('inf'))

        if self.key_indicator == 'auto':
            # Compare with previous best score directly
            if self.compare_func(key_score, best_score):
                best_score = key_score
                runner.meta['hook_msgs']['best_score'] = best_score
                runner.meta['hook_msgs']['best_ckpt'] = runner.save_checkpoint(
                    runner.work_dir,
                    filename_tmpl='best_{}.pth'.format(current))
                runner.logger.info(
                    f'Now best checkpoint is saved as best_{current}.pth.')
                runner.logger.info(
                    f'Best {self.key_indicator} is {best_score: .4f} '
                    f'at {cur_time} {cur_type}')
        else:
            if self.compare_func(key_score, best_score):
                best_score = key_score
                runner.meta['hook_msgs']['best_score'] = best_score
                runner.meta['hook_msgs']['best_ckpt'] = runner.save_checkpoint(
                    runner.work_dir,
                    filename_tmpl='best_{}.pth'.format(self.key_indicator))
                runner.logger.info(
                    f'Now best checkpoint is saved as '
                    f'best_{self.key_indicator}.pth.')
                runner.logger.info(
                    f'Best {self.key_indicator} is {best_score: .4f} '
                    f'at {cur_time} {cur_type}')


class DistEvalHook(_DistEvalHook):
    """Distributed EvalHook, with efficient test support.

    Args:
        by_epoch (bool): Determine perform evaluation by epoch or by iteration.
            If set to True, it will perform by epoch. Otherwise, by iteration.
            Default: False.
        efficient_test (bool): Whether save the results as local numpy files to
            save CPU memory during evaluation. Default: False.
    Returns:
        list: The prediction results.
    """

    greater_keys = ['mIoU', 'mAcc', 'aAcc']

    def __init__(self, *args, by_epoch=False, efficient_test=False, **kwargs):
        super().__init__(*args, by_epoch=by_epoch, **kwargs)
        self.efficient_test = efficient_test

    def _do_evaluate(self, runner):
        """perform evaluation and save ckpt."""
        # Synchronization of BatchNorm's buffer (running_mean
        # and running_var) is not supported in the DDP of pytorch,
        # which may cause the inconsistent performance of models in
        # different ranks, so we broadcast BatchNorm's buffers
        # of rank 0 to other ranks to avoid this.
        if self.broadcast_bn_buffer:
            model = runner.model
            for name, module in model.named_modules():
                if isinstance(module,
                              _BatchNorm) and module.track_running_stats:
                    dist.broadcast(module.running_var, 0)
                    dist.broadcast(module.running_mean, 0)

        if not self._should_evaluate(runner):
            return

        tmpdir = self.tmpdir
        if tmpdir is None:
            tmpdir = osp.join(runner.work_dir, '.eval_hook')

        from mmseg.apis import multi_gpu_test
        results = multi_gpu_test(
            runner.model,
            self.dataloader,
            tmpdir=tmpdir,
            gpu_collect=self.gpu_collect,
            efficient_test=self.efficient_test)
        if runner.rank == 0:
            print('\n')
            runner.log_buffer.output['eval_iter_num'] = len(self.dataloader)
            key_score = self.evaluate(runner, results)

            if self.save_best:
                self._save_ckpt(runner, key_score)

    def _save_ckpt(self, runner, key_score):
        """Save checkpoint with best score (distributed version).

        This override is needed because IterBasedRunner.save_checkpoint()
        doesn't accept 'save_best' parameter in newer MMCV versions.
        """
        if self.by_epoch:
            current = f'epoch_{runner.epoch + 1}'
            cur_type, cur_time = 'epoch', runner.epoch + 1
        else:
            current = f'iter_{runner.iter + 1}'
            cur_type, cur_time = 'iter', runner.iter + 1

        best_score = runner.meta.get('hook_msgs', {}).get(
            'best_score', self.key_indicator == 'auto' and 0 or -float('inf'))

        if self.key_indicator == 'auto':
            if self.compare_func(key_score, best_score):
                best_score = key_score
                runner.meta['hook_msgs']['best_score'] = best_score
                runner.meta['hook_msgs']['best_ckpt'] = runner.save_checkpoint(
                    runner.work_dir,
                    filename_tmpl='best_{}.pth'.format(current))
                runner.logger.info(
                    f'Now best checkpoint is saved as best_{current}.pth.')
                runner.logger.info(
                    f'Best {self.key_indicator} is {best_score: .4f} '
                    f'at {cur_time} {cur_type}')
        else:
            if self.compare_func(key_score, best_score):
                best_score = key_score
                runner.meta['hook_msgs']['best_score'] = best_score
                runner.meta['hook_msgs']['best_ckpt'] = runner.save_checkpoint(
                    runner.work_dir,
                    filename_tmpl='best_{}.pth'.format(self.key_indicator))
                runner.logger.info(
                    f'Now best checkpoint is saved as '
                    f'best_{self.key_indicator}.pth.')
                runner.logger.info(
                    f'Best {self.key_indicator} is {best_score: .4f} '
                    f'at {cur_time} {cur_type}')

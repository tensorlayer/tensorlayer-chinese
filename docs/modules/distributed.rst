API - 分布式
======================

分布式训练的帮助sessions和方法，请参考 `mnist例子 <https://github.com/zsdonghao/tensorlayer/blob/master/example/tutorial_mnist_distributed.py>`_。

.. automodule:: tensorlayer.distributed

.. autosummary::

   TaskSpecDef
   TaskSpec
   DistributedSession
   StopAtTimeHook
   LoadCheckpoint

分布式训练
----------------------

TaskSpecDef
^^^^^^^^^^^^^^^^^^^^^^

.. autofunction:: TaskSpecDef

Create TaskSpecDef from environment variables
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. autofunction:: TaskSpec

Distributed Session object
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. autofunction:: DistributedSession

Data sharding
^^^^^^^^^^^^^^^^^^^^^^

我们希望把数据分开很多块，放到每一个训练服务器上，而不是把整个数据放到所有的服务器上。
TensorFlow >= 1.4 提供了一些帮助类（helper classes）来支持数据分区功能（data sharding）: `Datasets <https://www.tensorflow.org/programmers_guide/datasets>`_

值得注意的是，在数据切分时，数据打乱非常重要，这些操作在建立shards的时候自动完成：

.. code-block:: python

  from tensorflow.contrib.data import TextLineDataset
  from tensorflow.contrib.data import Dataset

  task_spec = TaskSpec()
  files_dataset = Dataset.list_files(files_pattern)
  dataset = TextLineDataset(files_dataset)
  dataset = dataset.map(your_python_map_function, num_threads=4)
  if task_spec is not None:
        dataset = dataset.shard(task_spec.num_workers, task_spec.shard_index)
  dataset = dataset.shuffle(buffer_size)
  dataset = dataset.batch(batch_size)
  dataset = dataset.repeat(num_epochs)
  iterator = dataset.make_one_shot_iterator()
  next_element = iterator.get_next()
  with tf.device(task_spec.device_fn()):
        tensors = create_graph(next_element)
  with tl.DistributedSession(task_spec=task_spec,
                             checkpoint_dir='/tmp/ckpt') as session:
        while not session.should_stop():
            session.run(tensors)


Logging
^^^^^^^^^^^^^^^^^^^^^^

我们可以使用task_spec来对主服务器（master server）做日志记录：

.. code-block:: python

  while not session.should_stop():
        should_log = task_spec.is_master() and your_conditions
        if should_log:
            results = session.run(tensors_with_log_info)
            logging.info(...)
        else:
            results = session.run(tensors)

Continuous evaluation
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

我们可以使用其中一台子服务器（worker）来一直对保存下来对checkpoint做评估：

.. code-block:: python

  import tensorflow as tf
  from tensorflow.python.training import session_run_hook
  from tensorflow.python.training.monitored_session import SingularMonitoredSession

  class Evaluator(session_run_hook.SessionRunHook):
        def __init__(self, checkpoints_path, output_path):
            self.checkpoints_path = checkpoints_path
            self.summary_writer = tf.summary.FileWriter(output_path)
            self.lastest_checkpoint = ''

        def after_create_session(self, session, coord):
            checkpoint = tf.train.latest_checkpoint(self.checkpoints_path)
            # wait until a new check point is available
            while self.lastest_checkpoint == checkpoint:
                time.sleep(30)
                checkpoint = tf.train.latest_checkpoint(self.checkpoints_path)
            self.saver.restore(session, checkpoint)
            self.lastest_checkpoint = checkpoint

        def end(self, session):
            super(Evaluator, self).end(session)
            # save summaries
            step = int(self.lastest_checkpoint.split('-')[-1])
            self.summary_writer.add_summary(self.summary, step)

        def _create_graph():
            # your code to create the graph with the dataset

        def run_evaluation():
            with tf.Graph().as_default():
                summary_tensors = create_graph()
                self.saver = tf.train.Saver(var_list=tf_variables.trainable_variables())
                hooks = self.create_hooks()
                hooks.append(self)
                if self.max_time_secs and self.max_time_secs > 0:
                    hooks.append(StopAtTimeHook(self.max_time_secs))
                # this evaluation runs indefinitely, until the process is killed
                while True:
                    with SingularMonitoredSession(hooks=[self]) as session:
                        try:
                            while not sess.should_stop():
                                self.summary = session.run(summary_tensors)
                        except OutOfRangeError:
                            pass
                        # end of evaluation

  task_spec = TaskSpec().user_last_worker_as_evaluator()
  if task_spec.is_evaluator():
        Evaluator().run_evaluation()
  else:
        # run normal training



Session Hooks
----------------------

TensorFlow提供了一些 `Session Hooks <https://www.tensorflow.org/api_guides/python/train#Training_Hooks>`_
来对sessions做操作，我们在这里加更多的helper来实现更多的常规操作。

Stop after maximum time
^^^^^^^^^^^^^^^^^^^^^^

.. autofunction:: StopAtTimeHook

Initialize network with checkpoint
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. autofunction:: LoadCheckpoint

import tensorflow as tf



worker1 = "10.235.114.12:2222"
worker2 = "10.235.114.12:2223"
worker_hosts = [worker1, worker2]
cluster_spec = tf.train.ClusterSpec({ "worker": worker_hosts})
server = tf.train.Server(cluster_spec, job_name="worker", task_index=0)
server.join()
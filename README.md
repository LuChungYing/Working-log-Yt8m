# Working-log-Yt8m-ensamble

## step1 : 
    copy eval.py to TARGET
    
    paste   write_to_record()    &   get_output_feature()  to   TARGET
    
## step2 : 
#### buildgraph 是 eval.py       
#### inference 是 pre_infer_ensamble.py
#### get_input_data_tensors 是inference.py
    
    刪除 evaluate_loop()
    刪除以下
    ====================================
    fetches = {
        "video_id": tf.compat.v1.get_collection("video_id_batch")[0],
        "predictions": tf.compat.v1.get_collection("predictions")[0],
        "labels": tf.compat.v1.get_collection("labels")[0],
        "loss": tf.compat.v1.get_collection("loss")[0],
        "summary": tf.compat.v1.get_collection("summary_op")[0]
    }
    if FLAGS.segment_labels:
      fetches["label_weights"] = tf.compat.v1.get_collection("label_weights")[0]

    summary_writer = tf.compat.v1.summary.FileWriter(
        os.path.join(FLAGS.train_dir, "eval"),
        graph=tf.compat.v1.get_default_graph())

    evl_metrics = eval_util.EvaluationMetrics(reader.num_classes, FLAGS.top_k,
                                              None)

    last_global_step_val = -1
    while True:
      last_global_step_val = evaluation_loop(fetches, saver, summary_writer,
                                             evl_metrics, last_global_step_val)
      if FLAGS.run_once:
        break
    ==================================
   加入
       inference(saver,  FLAGS.model_checkpoint_path,  FLAGS.output_dir,   FLAGS.batch_size,   FLAGS.top_k)
       
  copy inference() in pre_infer_ensamble.py  to TARGET
     
  修改 調用inference( FLAGS.model_checkpoint_path 變成  FLAGS.train_dir)
  copy  get_input_data_tensors  in  inference.py   to TARGET
  把所有eval_data_pattern 改成input_data_pattern
  
  在 get_input_data_tensors() 裡加入
  labels_batch = input_data_dict["labels"]
    return video_id_batch, video_batch, num_frames_batch, labels_batch
    
   in inferece()
   把以下
   
     print model_checkpoint_path, FLAGS.train_dir
    if model_checkpoint_path is None:
       model_checkpoint_path = tf.train.latest_checkpoint(FLAGS.train_dir)
    print model_checkpoint_path, FLAGS.train_dir
    if model_checkpoint_path is None:
      raise Exception("unable to find a checkpoint at location: %s" % model_checkpoint_path)
   改成
   
    model_checkpoint_path = tf.train.latest_checkpoint(FLAGS.train_dir)

  在inference() 裡 的 try 下 加入從 inference.py copy的
         
       if FLAGS.segment_labels:
          results = get_segments(video_batch_val, num_frames_batch_val, 5)
          video_segment_ids = results["video_segment_ids"]
          video_id_batch_val = video_id_batch_val[video_segment_ids[:, 0]]
          video_id_batch_val = np.array([
              "%s:%d" % (x.decode("utf8"), y)
              for x, y in zip(video_id_batch_val, video_segment_ids[:, 1])
          ])
          video_batch_val = results["video_batch"]
          num_frames_batch_val = results["num_frames_batch"]
          if input_tensor.get_shape()[1] != video_batch_val.shape[1]:
            raise ValueError("max_frames mismatch. Please re-run the eval.py "
                             "with correct segment_labels settings.")
                             
      增加 video_batch_val <- video_batch       num_frames_batch_val  <-num_frames_batch
predictions_batch_val, video_id_batch_val, labels_batch_val, video_batch_val, num_frames_batch_val = sess.run([predictions_tensor, video_id_tensor, labels_tensor, video_batch, num_frames_batch])

在增加以下 讀取資料
video_id_batch, video_batch, num_frames_batch = get_input_data_tensors(
        reader, data_pattern, batch_size)
   
    
 

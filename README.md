# Working-log-Yt8m-ensamble

## step1 : 
    copy eval.py to TARGET
    
    paste   write_to_record()    &   get_output_feature()  to   TARGET
    
## step2 :
    eval_data_pattern ＝> input_data_pattern
    刪除 evaluate_loop()
    刪除以下（in  infer_pre_ensamble_test.py）
    ====================================
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
     

package edu.gatech.cse6250.feature_engineer

import edu.gatech.cse6250.model._
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.Dataset
import org.apache.spark.ml.feature.Tokenizer
import org.apache.spark.ml.feature.{ Word2Vec, Word2VecModel }

object FeatureEngineer {
  def GenerateWordMappers(input_dataset: Dataset[CleanedRow], input_col: String): (Map[String, Int], Map[String, org.apache.spark.ml.linalg.DenseVector]) = {
    val word2Vec = new Word2Vec().
      setInputCol(input_col).
      setVectorSize(100).
      setMinCount(0)
    val model = word2Vec.fit(input_dataset.select(input_col))
    val word2ind_mapper = model.getVectors.rdd.map(x => x(0)).zipWithIndex.map(x => (x._1.asInstanceOf[String], x._2.asInstanceOf[Int])).collect.toMap
    val word2vec_mapper = model.getVectors.rdd.map(x => (x(0), x(1))).collect.toMap.asInstanceOf[Map[String, org.apache.spark.ml.linalg.DenseVector]]
    (word2ind_mapper, word2vec_mapper)
  }

  def GenerateICD9CodeEncoder(input_CleanedRow_rdd: RDD[CleanedRow]): Map[String, Int] = {
    input_CleanedRow_rdd.flatMap(x => x.ICD9CODE).distinct.sortBy(x => x(0)).zipWithIndex.map(x => (x._1.asInstanceOf[String], x._2.asInstanceOf[Int])).collect.toMap
  }

  def GenerateOneHotVectorForWord(input_word: String, input_word2ind_mapper: Map[String, Int]): scala.collection.immutable.Vector[Int] = {
    val vocab_size = input_word2ind_mapper.size
    var return_vector = scala.collection.immutable.Vector.fill(vocab_size)(0)
    return_vector = return_vector.updated(input_word2ind_mapper(input_word), 1)
    return_vector
    // Archived
    // This uses Vector from spark mllib
    // var return_vector = Vectors.zeros(vocab_size)
    // return_vector.toArray(input_word2ind_mapper(input_word)) =
  }

  def GenerateFeatureRowRDD(input_CLEANEDROW_rdd: RDD[CleanedRow], input_icd9code2ind_mapper: Map[String, Int], input_word2ind_mapper: Map[String, Int], word2vec_mapper: Map[String, org.apache.spark.ml.linalg.DenseVector]): RDD[FeatureRow] = {
    val return_rdd = input_CLEANEDROW_rdd.map { row =>
      // Only keep ICD9code that exist in the
      val filtered_icd9code = row.ICD9CODE.filter(code => input_icd9code2ind_mapper.keys.toList.contains(code))

      // Generate label-encoded icd9code (integer id for icd9code)
      val filtered_icd9code_idx = filtered_icd9code.map(code => input_icd9code2ind_mapper(code))

      // Generate icd9code vector
      var filtered_icd9code_vector = scala.collection.immutable.Vector.fill(input_icd9code2ind_mapper.keys.size)(0)
      filtered_icd9code_idx.foreach { i =>
        filtered_icd9code_vector = filtered_icd9code_vector.updated(i, 1)
      }

      // Only keep tokens that exist in the training set
      val filtered_note_tokens = row.CLEANED_TOKENS.filter(token => input_word2ind_mapper.keys.toList.contains(token))

      val filtered_note_tokens_idx = filtered_note_tokens.map(token => input_word2ind_mapper(token))

      // Generate OH_TOKENS and W2V_TOKENS
      var onehot_array = Array[scala.collection.immutable.Vector[Int]]()
      var word2vec_array = Array[org.apache.spark.ml.linalg.DenseVector]()
      filtered_note_tokens.zipWithIndex.foreach {
        case (each_word, i) =>
          val one_oh_vector = GenerateOneHotVectorForWord(each_word, input_word2ind_mapper)
          onehot_array ++= Array(one_oh_vector) // Reference: https://alvinalexander.com/scala/how-to-create-multidimensional-arrays-in-scala-cookbook/
          val one_w2v_vector = word2vec_mapper(each_word)
          word2vec_array ++= Array(one_w2v_vector)
      }
      FeatureRow(row.SUBJECT_ID, row.HADM_ID, row.ICD9CODE, filtered_icd9code_idx, filtered_icd9code_vector, filtered_note_tokens, filtered_note_tokens_idx, onehot_array, word2vec_array)
    }
    return_rdd
  }
}
/**
 * @author
 * @author
 */

package edu.gatech.cse6250.main

import java.text.SimpleDateFormat

import edu.gatech.cse6250.helper.{ CSVHelper, SparkHelper, utils }
import edu.gatech.cse6250.model._
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.SparkSession
import org.apache.spark.sql.functions.{ expr, col, column }
import org.apache.spark.sql.Dataset

import org.apache.spark.mllib.linalg.{ Vector, Vectors }

import edu.gatech.cse6250.feature_engineer.FeatureEngineer

object Main {
  def main(args: Array[String]) {
    import org.apache.log4j.{ Level, Logger }

    Logger.getLogger("org").setLevel(Level.WARN)
    Logger.getLogger("akka").setLevel(Level.WARN)

    val spark = SparkHelper.spark
    import spark.implicits._
    val sc = spark.sparkContext
    val sqlContext = spark.sqlContext

    // Import data from DIAGNOSES_ICD.csv
    // Choose only rows with
    //  ICD9_CODE IS NOT NULL
    val diagnoses_dataset = spark.read.format("com.databricks.spark.csv").
      option("header", "true").
      option("mode", "DROPMALFORMED").
      option("delimiter", ",").
      load("hdfs://bootcamp.local:9000/project/data/mimiciii/DIAGNOSES_ICD.csv").
      filter("ICD9_CODE IS NOT NULL")
    // filter(col("SUBJECT_ID") === 9 || col("SUBJECT_ID") === 13)
    diagnoses_dataset.createOrReplaceTempView("diagnoses")

    // Import data from NOTEEVENTS.csv
    // Choose only rows with
    //  Category = Discharge summary
    //  ISERROR is null
    val noteevents_dataset = spark.read.format("com.databricks.spark.csv").
      option("header", "true").
      option("mode", "DROPMALFORMED").
      option("delimiter", ",").
      option("multiline", true).
      load("hdfs://bootcamp.local:9000/project/data/mimiciii/NOTEEVENTS.csv").
      filter(col("CATEGORY") === "Discharge summary").
      filter("ISERROR IS NULL").
      filter("TEXT IS NOT NULL").
      filter(col("TEXT") =!= "")
    // filter(col("SUBJECT_ID") === 9 || col("SUBJECT_ID") === 13)
    // filter("SUBJECT_ID IN ('9', '13')")
    noteevents_dataset.createOrReplaceTempView("noteevents")

    // Generate modified_diagnoses_dataset with 3 columns: SUBJECT_ID, HADM_ID, ICD9CODE
    // ICD9CODE is a List of ICD9_CODE associated with corresponding (SUBJECT_ID, HADM_ID)
    sqlContext.udf.register("reformat_icd9code", (code: String, is_diag: Boolean) => utils.reformat_icd9code(code, is_diag))
    val modified_diagnoses_dataset = sqlContext.sql("""
      SELECT SUBJECT_ID, HADM_ID, COLLECT_LIST(ICD9CODE) ICD9CODE
      FROM(
        SELECT SUBJECT_ID, HADM_ID, reformat_icd9code(ICD9_CODE, true) ICD9CODE
        FROM diagnoses
        ORDER BY SUBJECT_ID, HADM_ID, ICD9_CODE
      )
      GROUP BY SUBJECT_ID, HADM_ID
    """)
    modified_diagnoses_dataset.createOrReplaceTempView("modified_diagnoses")

    // Generate modified_noteevents_dataset with 3 columns: SUBJECT_ID, HADM_ID, TEXT
    // TEXT is a concatenated string of all the notes of the corresponding (SUBJECT_ID, HADM_ID)
    val modified_noteevents_dataset = sqlContext.sql("""
      SELECT SUBJECT_ID, HADM_ID, CONCAT_WS(" ", COLLECT_LIST(TEXT)) TEXT
      FROM(
          SELECT SUBJECT_ID, HADM_ID, DATE(CHARTDATE) CHARTDATE, TEXT 
          FROM noteevents
          ORDER BY SUBJECT_ID, HADM_ID, CHARTDATE
      )
      GROUP BY SUBJECT_ID, HADM_ID
      ORDER BY SUBJECT_ID, HADM_ID
    """)
    modified_noteevents_dataset.createOrReplaceTempView("modified_noteevents")

    // Generate merged_dataset
    // to inner-join modified_diagnoses_dataset and modified_noteevents_dataset
    val merged_dataset = sqlContext.sql("""
      SELECT a.SUBJECT_ID, a.HADM_ID, a.ICD9CODE, b.TEXT
      FROM modified_diagnoses a
      INNER JOIN modified_noteevents b
        ON a.SUBJECT_ID = b.SUBJECT_ID AND a.HADM_ID = b.HADM_ID
      order by a.SUBJECT_ID, a.HADM_ID
    """)

    // merged_dataset.write.format("csv").mode("overwrite").save("hdfs://bootcamp.local:9000/project/data/")
    // merged_dataset.write.format("com.databricks.spark.csv").mode("overwrite").save("./merged_dataset.csv")

    // Text-processing notes
    import org.apache.spark.ml.feature.RegexTokenizer
    val rt = new RegexTokenizer().
      setInputCol("TEXT").
      setOutputCol("TOKENS").
      setPattern("\\w+").setGaps(false).
      setToLowercase(true)
    val merged_dataset_with_tokens = rt.transform(merged_dataset)

    import org.apache.spark.ml.feature.StopWordsRemover
    val english_stopwords = StopWordsRemover.loadDefaultStopWords("english")
    val stop_word_remover = new StopWordsRemover().
      setStopWords(english_stopwords).
      setInputCol("TOKENS").
      setOutputCol("TOKENS_WITHOHUT_STOPWORDS")
    val merged_dataset_with_tokens_without_stopwords = stop_word_remover.transform(merged_dataset_with_tokens)

    val merged_dataset_with_tokens_without_stopwords_rdd = merged_dataset_with_tokens_without_stopwords.rdd

    // Remove all tokens that are entirely numeric, e.g. remove 500 but keep 500mg
    // Fit each row of data into case class CleanedRow
    val CleanedRow_rdd: RDD[CleanedRow] = merged_dataset_with_tokens_without_stopwords_rdd.map { x =>
      {
        CleanedRow(x(0).toString, x(1).toString, x.getSeq[String](2).toList, x(3).toString, x.getSeq[String](5).toList.filter(x => x.forall(_.isDigit) == false))
      }
    }
    // CleanedRow_rdd.saveAsObjectFile("hdfs://bootcamp.local:9000/project/data/processed/CleanedRow_rdd")

    val (train_hadm_id, test_hadm_id, val_hadm_id) = utils.train_test_val_split(CleanedRow_rdd.map(x => x.HADM_ID).distinct.collect.toList)
    val train_CleanedRow_rdd = CleanedRow_rdd.filter(x => train_hadm_id.contains(x.HADM_ID))
    val test_CleanedRow_rdd = CleanedRow_rdd.filter(x => test_hadm_id.contains(x.HADM_ID))
    val val_CleanedRow_rdd = CleanedRow_rdd.filter(x => val_hadm_id.contains(x.HADM_ID))

    // Obtain word2ind_mapper and word2vec_mapper
    // Important: word2ind_mapper and word2vec_mapper must be generated from train data only
    val (word2ind_mapper, word2vec_mapper) = FeatureEngineer.GenerateWordMappers(spark.createDataset(train_CleanedRow_rdd), "CLEANED_TOKENS")

    // Obtain label encoder for ICD9Code, i.e. icd9code2ind_mapper
    // Important: incd9code2ind_mapper must be generated from train data only
    val icd9code2ind_mapper = FeatureEngineer.GenerateICD9CodeEncoder(train_CleanedRow_rdd)

    val train_FeatureRow_rdd = FeatureEngineer.GenerateFeatureRowRDD(train_CleanedRow_rdd, icd9code2ind_mapper, word2ind_mapper, word2vec_mapper)
    val test_FeatureRow_rdd = FeatureEngineer.GenerateFeatureRowRDD(test_CleanedRow_rdd, icd9code2ind_mapper, word2ind_mapper, word2vec_mapper)
    val val_FeatureRow_rdd = FeatureEngineer.GenerateFeatureRowRDD(val_CleanedRow_rdd, icd9code2ind_mapper, word2ind_mapper, word2vec_mapper)

    // val TRAIN_FILE_PATH: String = "hdfs://bootcamp.local:9000/project/data/processed/train_FeatureRow_rdd.json"
    // val TEST_FILE_PATH: String = "hdfs://bootcamp.local:9000/project/data/processed/test_FeatureRow_rdd.json"
    // val VAL_FILE_PATH: String = "hdfs://bootcamp.local:9000/project/data/processed/val_FeatureRow_rdd.json"

    val TRAIN_FILE_PATH: String = "file:/mnt/host/home/gia/Documents/OMSCS/Project/ProcessedData/train_FeatureRow_rdd.json"
    val TEST_FILE_PATH: String = "file:/mnt/host/home/gia/Documents/OMSCS/Project/ProcessedData/test_FeatureRow_rdd.json"
    val VAL_FILE_PATH: String = "file:/mnt/host/home/gia/Documents/OMSCS/Project/ProcessedData/val_FeatureRow_rdd.json"

    spark.createDataFrame(train_FeatureRow_rdd).write.json(TRAIN_FILE_PATH)
    spark.createDataFrame(test_FeatureRow_rdd).write.json(TEST_FILE_PATH)
    spark.createDataFrame(val_FeatureRow_rdd).write.json(VAL_FILE_PATH)

    sc.stop()
  }
}


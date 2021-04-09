/**
 * @author
 */

package edu.gatech.cse6250.model

case class Admission(patientID: String, HADM_ID: String)

case class Diagnosis(patientID: String, HADM_ID: String, ICD9_Code: String)

case class NoteEvent(patientID: String, HADM_ID: String, CATEGORY: String, DESCRIPTION: String, CGID: String, TEXT: String)

case class CleanedRow(SUBJECT_ID: String, HADM_ID: String, ICD9CODE: List[String], TEXT: String, CLEANED_TOKENS: List[String])

case class FeatureRow(SUBJECT_ID: String, HADM_ID: String, ICD9CODE: List[String], ICD9CODE_IDX: List[Int], ICD9CODE_VECTOR: scala.collection.immutable.Vector[Int], CLEANED_TOKENS: List[String], CLEANED_TOKENS_IDX: List[Int], OH_TOKENS: Array[scala.collection.immutable.Vector[Int]], W2V_TOKENS: Array[org.apache.spark.ml.linalg.DenseVector])
package edu.gatech.cse6250.helper

object utils {
  def reformat_icd9code(code: String, is_diag: Boolean): String = {
    var return_code = code
    if (is_diag == true) {
      if (code.startsWith("E")) {
        if (code.length() > 4) {
          return_code = code.substring(0, 4).concat(".").concat(code.substring(4))
        }
      } else {
        if (code.length() > 3) {
          return_code = code.substring(0, 3).concat(".").concat(code.substring(3))
        }
      }
    } else {
      return_code = code.substring(0, 2).concat(".").concat(code.substring(2))
    }
    return_code
  }

  def train_test_val_split(input_list: List[String]) = {
    // val shuffled_list = scala.util.Random.setSeed(100).shuffle(input_list)
    scala.util.Random.setSeed(100)

    val shuffled_list = scala.util.Random.shuffle(input_list)

    val train_list = shuffled_list.slice(0, (0.7 * shuffled_list.size).toInt)

    val test_list = shuffled_list.slice((0.7 * shuffled_list.size).toInt, (0.85 * shuffled_list.size).toInt)

    val val_list = shuffled_list.slice((0.85 * shuffled_list.size).toInt, shuffled_list.size)

    (train_list, test_list, val_list)
  }
}
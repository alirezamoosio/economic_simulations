package bo

import java.nio.file.{Files, Paths}

import Simulation.Simulation
import breeze.linalg.{DenseMatrix, DenseVector, linspace, normalize}
import breeze.plot.{Figure, plot}
import org.apache.commons.csv.{CSVFormat, CSVParser, CSVPrinter}

case class Viz(f: Simulation => Seq[Double], outputNames: Array[String], var params: Map[String, Double]) {

  def plotSimOverParam(param: String,
                       bounds: (Double, Double),
                       numberOfPoints: Int = 100,
                       runSimTill: Int = 1000): Unit = {
    val csvFilePath = s"results/csv/plotsOver$param.csv"
    val (x: DenseVector[Double], ys: DenseMatrix[Double]) =
      if (Files.exists(Paths.get(csvFilePath))) {
        readCsvFile(csvFilePath)
      } else {
        val x = linspace(bounds._1, bounds._2, numberOfPoints)
        val ys = DenseMatrix(
          x.map(value => {
            println(s"running sim for param $value")
            params += param -> value
            val s = new Simulation(params)
            Main.initializeSimulation(s)
            Main.callSimulation(s, runSimTill)
            f(s)
          }).toArray:_*
        )
        writeCsvFile(x, ys, csvFilePath)
        (x, ys)
      }

    val figureSingle = Figure("Single Measures")
    val pSingle = figureSingle.subplot(0)
    for (i <- 0 until ys.cols) {
      val normalized = normalize(ys(::, i)).map(_ * 100)
      pSingle += plot(x, normalized, name = outputNames(i))
    }
    pSingle.xlabel = s"$param"
    pSingle.ylabel = "f"
    pSingle.setXAxisDecimalTickUnits()
    pSingle.legend = true
    try {
      figureSingle.saveas(s"results/singlePlotsOver$param.png")
    } catch {
      case _: Exception => // ignore
    }

    val figureSum = Figure("Sum of Measures")
    val pSum = figureSum.subplot(0)
    val normalized: Seq[DenseVector[Double]] = for (i <- 0 until ys.cols) yield normalize(ys(::, i)).map(_ * 100)
    pSum += plot(x, normalized.foldLeft(DenseVector.zeros[Double](normalized.head.size))(_ + _))
    pSum.xlabel = s"$param"
    pSum.ylabel = "f"
    pSum.setXAxisDecimalTickUnits()
    figureSum.saveas(s"results/sumPlotOver$param.png")
  }

  def plotSimOverTime(bounds: (Int, Int),
                      numberOfPoints: Int = 1000): Unit = {
    val t = linspace(bounds._1, bounds._2, numberOfPoints)
    val step = (bounds._2 - bounds._1) / numberOfPoints
    assert(step > 0, "step size must be a positive non-zero integer!")
    val s = new Simulation(params)
    Main.initializeSimulation(s)
    val y = t.map(_ => {
      Main.callSimulation(s, step)
      f(s).head
    }).toDenseVector

    val figure = Figure()
    val p = figure.subplot(0)
    p += plot(t, y)
    p.xlabel = "time"
    p.ylabel = "f"
    figure.saveas("results/plotOverTime.png")
  }

  private def readCsvFile(path: String): (DenseVector[Double], DenseMatrix[Double]) = {
    var x: Array[Double] = Array()
    var ys: Array[Array[Double]] = Array()
    try {
      val reader = Files.newBufferedReader(Paths.get(path))
      val csvParser = new CSVParser(reader, CSVFormat.DEFAULT.withFirstRecordAsHeader.withIgnoreHeaderCase.withTrim)
      try {
        val itererator = csvParser.iterator()
        while (itererator.hasNext) {
          val csvRecord = itererator.next()
          x :+= csvRecord.get("x").toDouble
          var y = Array[Double]()
          for (name <- outputNames) {
            y :+= csvRecord.get(name).toDouble
          }
          ys :+= y
        }
      } finally {
        if (reader != null) reader.close()
        if (csvParser != null) csvParser.close()
      }
    }
    (DenseVector(x), DenseMatrix(ys:_*))
  }

  private def writeCsvFile(x: DenseVector[Double], ys: DenseMatrix[Double], path: String) = {
    try {
      val writer = Files.newBufferedWriter(Paths.get(path))
      val csvPrinter = new CSVPrinter(writer, CSVFormat.DEFAULT.withHeader((Array("x") ++ outputNames):_*))
      try {
        for (i <- 0 until x.length) {
          val array: Array[String] = x(i).toString +: ys(i, ::).inner.map(_.toString).toArray
          csvPrinter.printRecord(array:_*)
        }
        csvPrinter.flush()
      } finally {
        if (writer != null) writer.close()
        if (csvPrinter != null) csvPrinter.close()
      }
    }
  }
}

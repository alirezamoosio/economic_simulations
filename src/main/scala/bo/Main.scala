package bo

import java.io.{BufferedOutputStream, FileDescriptor, FileOutputStream, PrintStream}

import Securities.Land
import Simulation.{Person, Simulation}
import _root_.Simulation.SimLib._
import bo.DatasetCreator.{Data, Statistics}
import breeze.stats.distributions.Gaussian
import spray.json.JsonParser

import scala.collection.mutable.{Map => MutableMap}

object Main {
  val bufferSize = math.pow(2, 27).toInt // 128 MB
  val initLog = new BufferedOutputStream(new FileOutputStream("target/scala-2.11/initLog"), bufferSize)
  val runLog = new BufferedOutputStream(new FileOutputStream("target/scala-2.11/runLog"), bufferSize)

  /**
    *
    * @return a sequence of triples indexed over time, where the first element of the triple is input data,
    *         the second in the output data and the third is the global statistics
    */
  def simFunction(constants: Data, variables: Data,
                  nSteps: Int, stepSize: Int, agents: Iterable[String]): Seq[(Data, Data, Statistics)] = {
    val s = new Simulation(constants, variables)
    Main.initializeSimulation(s)
    var data_out = s.getPopulationData(agents)
    for (_ <- 1 to nSteps) yield {
      val data_in = data_out
      Main.runSimulation(s, stepSize)
      data_out = s.getPopulationData(agents)
      (data_in, s.getPopulationData(agents), s.getGlobalStat)
    }
  }

  def main(args: Array[String]): Unit = {
    val source = scala.io.Source.fromFile(args(1))
    val jsonString = source.mkString
    source.close()

    val jsonAst = JsonParser(jsonString)
    val rawConstants = jsonAst.asJsObject.fields("constants")
    val rawVariables = jsonAst.asJsObject.fields("variables")
    val constants = scala.collection.mutable.Map(
      rawConstants.asJsObject.fields.mapValues(_.asJsObject.fields.mapValues(_.toString.toDouble)).toSeq: _*)
    val variables = scala.collection.mutable.Map(
      rawVariables.asJsObject.fields.mapValues(_.asJsObject.fields.mapValues(_.toString.toDouble)).toSeq: _*)


    args(0) match {
      case "generate" =>
        val nSamples = args(2).toInt
        val nSteps = args(3).toInt
        val stepSize = if (args(4).forall(_.isDigit)) args(4).toInt else 10
        var agents = if (args(4).forall(_.isDigit)) args.slice(5, args.length) else args.slice(4, args.length)
        agents = if (agents.length == 1 && agents.head.equals("all")) GLOBAL.allAgents else agents
        DatasetCreator.createDataset(nSamples, agents, nSteps, stepSize)

      case "evaluate" =>
        val stepSize = args(2).toInt
        val entry = args(3).toInt
        val (matrix, header) = CsvManager.readCsvFile("target/scala-2.11/global_stats.csv")
        val actuals: Statistics = header.toArray.zip(matrix(entry, ::).inner.toArray).toMap
        println(Metrics.meanAbsoluteError(simFunction(
          constants, variables, 1, stepSize, GLOBAL.allAgents).map(_._3).last, actuals))

      case "lyapunov" =>
        println(ChaosTest(constants, variables).lyapunovExponent(args(2), args(3)))

      case "plot-time" =>
        val visualizer = Viz(constants, variables)
        val from = if (args.length > 2) args(2).toInt else 0
        val to = if (args.length > 3) args(3).toInt else 300
        val points = if (args.length > 4) args(4).toInt else 300
        visualizer.plotSimOverTime((from, to), points)

      case "plot-param" =>
        val visualizer = Viz(constants, variables)
        val from = if (args.length > 4) args(4).toDouble else 0
        val to = if (args.length > 5) args(5).toDouble else 100
        val simIters = if (args.length > 6) args(6).toInt else 300
        visualizer.plotSimOverParam(args(2), args(3), (from, to), runSimTill = simIters)
    }

    initLog.flush()
    runLog.flush()
  }

  def initializeSimulation(s: Simulation, mute: Boolean = true): Unit = {
    if (mute)
      Console.setOut(initLog)

    val landlord = new Source(Land, 20, 100000 * 100, s)

    val genderDistr = Gaussian(s.constants("Person")("genderMu"), s.constants("Person")("genderSigma"))
    val people = for (_ <- 1 to s.constants("Person")("number").toInt) yield {
      val male = math.max(0, math.min(1, genderDistr.sample())) <= 0.5
      new Person(s, true, male)
    }

    val farms = for (_ <- 1 to s.constants("Farm")("number").toInt) yield new Farm(s)
    val mills = for (_ <- 1 to s.constants("Mill")("number").toInt) yield new Mill(s)
    val bakeries = for (_ <- 1 to s.constants("Bakery")("number").toInt) yield new Bakery(s)
    val cattleFarms = for (_ <- 1 to s.constants("CattleFarm")("number").toInt) yield new CattleFarm(s)
    val oilFields = for (_ <- 1 to s.constants("OilField")("number").toInt) yield new OilField(s)
    val refineries = for (_ <- 1 to s.constants("Refinery")("number").toInt) yield new Refinery(s)

    val sims = farms ++ mills ++ bakeries ++ cattleFarms ++ oilFields ++ refineries ++ people

    s.init(List(
      landlord
    ) ++ sims.toList)

    Console.out.flush()
    Console.setOut(new PrintStream(new FileOutputStream(FileDescriptor.out)))
  }

  def runSimulation(s: Simulation, iterations: Int, mute: Boolean = true): Unit = {
    if (mute)
      Console.setOut(runLog)

    s.run(iterations)

    Console.out.flush()
    Console.setOut(new PrintStream(new FileOutputStream(FileDescriptor.out)))
  }
}

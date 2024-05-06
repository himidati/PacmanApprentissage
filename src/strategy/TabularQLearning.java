package strategy;

import java.util.ArrayList;
import java.util.Collection;
import java.util.Collections;
import java.util.Random;

import org.nd4j.nativeblas.Nd4jCpu.expose;

import agent.Agent;
import agent.AgentAction;
import agent.PositionAgent;
import motor.Maze;
import motor.PacmanGame;
import neuralNetwork.TrainExample;

import java.util.HashMap;


public class TabularQLearning  extends QLearningStrategy{


	HashMap<String, double[]> QTable;



	int sizeMazeX;
	int sizeMazeY;




	public TabularQLearning( double epsilon, double gamma, double alpha,  int sizeMazeX, int sizeMazeY, int nbWalls) {
		
		super( epsilon, gamma, alpha, sizeMazeX, sizeMazeY);

		this.sizeMazeX = sizeMazeX;
		this.sizeMazeY = sizeMazeY;

		System.out.println("sizeX labyrinth " + this.sizeMazeX);
		System.out.println("sizeY labyrinth " + this.sizeMazeY);
		
		int numberCellsWithoutWall = sizeMazeX*sizeMazeY - nbWalls;
				
		System.out.println("NumberCells without wall " + numberCellsWithoutWall);

		int numberStates =  (int) Math.pow( 4, numberCellsWithoutWall);

		System.out.println("Max number different states " + numberStates);

		QTable = new HashMap<>();


	}

	/** 
	 * encode l'état du jeu actuelle sous forme de chaine de caractère
	 * @param state
	 * @return
	 */
	public String encodeState(PacmanGame state){	
		String encodeState="";		
		for(int i=0; i<this.sizeMazeX;i++){
			for(int j=0; j<this.sizeMazeY; j++){
				if (!state.isWallAtPosition(i, j)) {
					if (state.isCapsuleAtPosition(i, j)) {
						encodeState += "1";
					} else if (state.isPacmanAtPosition(i, j)) {
						encodeState += "2";
					} else if (state.isGhostAtPosition(i, j)) {
						encodeState += "3";
					} else if (state.isGumAtPosition(i, j)) {
						encodeState += "4";
					} else {
						encodeState += "0";
					} //encoder lorsque les fantômes sont appeuré, conteur du nombre de tours restants depuis qu'on a mangé la pacgomme
				}
			}			
		}

		return encodeState;
	}


	@Override
	public synchronized AgentAction chooseAction(PacmanGame state) {

		ArrayList<AgentAction> actionsPossible=state.getLegalPacmanActions();

		AgentAction actionChoisi=actionsPossible.get(0); //par défaut on choisie la première actions possible
	
		String etatencodee=encodeState(state);

		if (Math.random() < this.current_epsilon) { // action aléatoire parmis les actions possible
				int indexAleatoire = (int) (Math.random() * actionsPossible.size());
				actionChoisi = actionsPossible.get(indexAleatoire);
			} else if (this.QTable.containsKey(etatencodee)) { // si la clé existe, on choisi la meilleur action 
					double[] valeureQ=this.QTable.get(etatencodee); //récupère les valeurs de l'etat déjà présent

					if(valeureQ[0]>valeureQ[1] && valeureQ[0]>valeureQ[2] && valeureQ[0]>valeureQ[3]){
						return new AgentAction(AgentAction.NORTH);
					}else if(valeureQ[1]>valeureQ[0] && valeureQ[1]>valeureQ[2] && valeureQ[1]>valeureQ[3]) 
					return new AgentAction(AgentAction.SOUTH);
					else if(valeureQ[2]>valeureQ[0] && valeureQ[2]>valeureQ[1] && valeureQ[2]>valeureQ[3])
					return new AgentAction(AgentAction.EAST);
					else if(valeureQ[3]>valeureQ[0] && valeureQ[3]>valeureQ[1] && valeureQ[3]>valeureQ[2])
					return new AgentAction(AgentAction.WEST);
					else actionChoisi=actionsPossible.get((int)Math.random()*actionsPossible.size());

			} else { // si la clé n'existe pas on crée un tableau de double de taille 4 avec des valeurs = 0 de Qtable
					//et on choisie une action aléatoire parmis les actions possibles
				double[] newQValeurs = new double[4];
				this.QTable.put(encodeState(state), newQValeurs);
				int index=(int)Math.random()*actionsPossible.size();
				actionChoisi=actionsPossible.get(index);
			}

		return actionChoisi;

	}

	private double getMaxQValue(double[] qValues) {
		double maxQValue = qValues[0];
		for (int i = 0; i < qValues.length; i++) {
			if (qValues[i] > maxQValue) {
				maxQValue = qValues[i];
			}
		}
		return maxQValue;
	}
	
	@Override
	public void update(PacmanGame state, PacmanGame nextState, AgentAction action, double reward,boolean isFinalState) {
		
		String encodageCurrentState = encodeState(state);
		String encodageNextState = encodeState(nextState);

		// mis à jour de la valeur Q pour l'action choisie à l'etat actuel
		double[] valQCurrentState = this.QTable.getOrDefault(encodageCurrentState, new double[4]);

		// on calcule la meilleurs valeurs Q pour l'état suivant
		double bestNextActionValue = Double.NEGATIVE_INFINITY;
		double[] valQNextSate = this.QTable.getOrDefault(encodageNextState, new double[4]);
		bestNextActionValue = getMaxQValue(valQNextSate);

		double oldQvalue = getMaxQValue(valQCurrentState);

		double newQvalue = (1 - this.learningRate) * oldQvalue
				+ this.learningRate * (reward + this.gamma * bestNextActionValue);
		valQCurrentState[action.get_idAction()] = newQvalue;

		this.QTable.replace(encodageCurrentState, valQCurrentState);

	}

	@Override
	public void learn(ArrayList<TrainExample> trainExamples) {
		for(TrainExample example : trainExamples){
			
		}
		
	}

}

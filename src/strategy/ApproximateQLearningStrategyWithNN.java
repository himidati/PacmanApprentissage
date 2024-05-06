package strategy;

import java.util.ArrayList;
import java.util.List;
import java.util.Random;

import agent.Agent;
import agent.AgentAction;

import agent.PositionAgent;

import motor.PacmanGame;
import neuralNetwork.NeuralNetWorkDL4J;

import neuralNetwork.TrainExample;



public class ApproximateQLearningStrategyWithNN extends QLearningStrategy {

	int d=3;
	NeuralNetWorkDL4J nn;

	int nEpochs;
	int batchSize;
	double baseEpsilon;
	//double [] table;
	Random random= new Random();


	public ApproximateQLearningStrategyWithNN(double epsilon, double gamma, double learningRate,  int nEpochs, int batchSize, int sizeMazeX , int sizeMazeY) {

		super(epsilon, gamma, learningRate, sizeMazeX, sizeMazeY);

		this.nEpochs = nEpochs;
		this.batchSize = batchSize;
		//this.table=new double [];
		this.nn=new NeuralNetWorkDL4J(this.learningRate, 0, this.d+1,1);
	}


	/**
	 * renvoie vrai si des fantômes se trouvent dans une case au tour du pacman
	 * après avoir joué
	 * l'action action à l'état state (on suppose que les fantômes ne bougent pas
	 * entre eux)
	 */
	public boolean isNextFantome(PacmanGame state, AgentAction action) {
		int posXPac = state.getPacmanX();
		int posYPac = state.getPacmanY();

		// Positions adjacentes au Pacman en fonction de l'action
		int[][] adjacentPositions = {
				{ posXPac, posYPac - 1 }, // Nord
				{ posXPac, posYPac + 1 }, // Sud
				{ posXPac + 1, posYPac }, // Est
				{ posXPac - 1, posYPac }, // Ouest
				{ posXPac + 1, posYPac - 1 }, // Nord-Est
				{ posXPac - 1, posYPac - 1 }, // Nord-Ouest
				{ posXPac + 1, posYPac + 1 }, // Sud-Est
				{ posXPac - 1, posYPac + 1 } // Sud-Ouest
		};

		// Vérifie la présence de fantômes dans les positions adjacentes
		for (int[] pos : adjacentPositions) {
			int x = pos[0];
			int y = pos[1];

			// vérifie que les positions restent dans les limites du labyrinthe
			if (x < 0)
				x = this.sizeMazeX - 1;
			if (x >= this.sizeMazeX)
				x = 0;
			if (y < 0)
				y = this.sizeMazeY - 1;
			if (y >= this.sizeMazeY)
				y = 0;

			// Vérifie la présence de fantômes dans la position adjacente
			if (state.isGhostAtPosition(x, y)) {
				return true;
			}
		}

		return false;
	}

	/**
	 * renvoie true si le pacman peut manger une pacgomme en effectuant l'action
	 * action à l'etat state sinon false
	 */
	public boolean canEat(PacmanGame state, AgentAction action) {

		int posXPac = state.getPacmanX();
		int posYPac = state.getPacmanY();

		switch (action.get_idAction()) {
			case 0:
				if (posYPac > 0) posYPac -= 1;  else posYPac=this.sizeMazeY-1;
				break;
			case 1:
				posYPac=(posYPac+1);
				break;
			case 2:
				posXPac=(posXPac+1);
				break;
			case 3:
				if (posXPac > 0) posXPac -= 1; else posXPac=this.sizeMazeX-1;
				break;
		}
		return state.isGumAtPosition(posXPac, posYPac);
	}

	public static int getDistance(int x1, int y1, int x2, int y2) {
		int dx = Math.abs(x2 - x1);
		int dy = Math.abs(y2 - y1);
		return dx + dy;
	}

	/**
	 * retourne le nombre de coup qu'on va jouer pour atteindre le pacgomme le plus
	 * proche apès avoir joué l'action action à l'état state
	 */
	public double nbCoup(PacmanGame state, AgentAction action) {
		double nbcoups=0.0;

		int posXPac = state.getPacmanX();
		int posYPac = state.getPacmanY();


		if(state.isGumAtPosition(posXPac, posYPac)) ++nbcoups;
		switch (action.get_idAction()) {
			case 0:
				if (posYPac > 0) posYPac -= 1;  else posYPac=this.sizeMazeY-1;
				if(state.isGumAtPosition(posXPac, posYPac)) ++nbcoups;
				break;
			case 1:
				posYPac=(posYPac+1)% this.sizeMazeY;
				if(state.isGumAtPosition(posXPac, posYPac)) ++nbcoups;
				break;
			case 2:
				posXPac=(posXPac+1)% this.sizeMazeX;
				if(state.isGumAtPosition(posXPac, posYPac)) ++nbcoups;
				break;
			case 3:
				if (posXPac > 0) posXPac -= 1; else posXPac=this.sizeMazeX-1;
				if(state.isGumAtPosition(posXPac, posYPac)) ++nbcoups;
				break;
		}

		//récupère la liste des positions des pacgommes
		List<int[]> posPacgommes = new ArrayList<>();
		for(int i =0; i < this.sizeMazeX;i++) {
			for(int j =0; j <this.sizeMazeY; j++) {
				if(state.isGumAtPosition(i, j)); posPacgommes.add(new int[]{i,j});
			}
		}
		double distMin=Double.MAX_VALUE;
		for (int i=0; i<posPacgommes.size(); i++) {
			int distance=getDistance(posXPac, posYPac, posPacgommes.get(i)[0], posPacgommes.get(i)[1]);
			if(distance<distMin){
				distMin=distance;
			}
		}
		nbcoups+=distMin;
		return nbcoups;
	}


	public double[] extractFeatures(PacmanGame state, AgentAction action ){
		double[] f =new double[d+1];

		f[0]=1;
		f[1]=canEat(state, action) ? 12.0 : 0.0;
		f[2]=(nbCoup(state, action)<3 && nbCoup(state, action)!=0)? 12 : 0.0;
		f[3]=isNextFantome(state, action)? -12.0 : 0.0;
		
		return f;
	}

	/**
	 * return true lorsqu'une action fait partie des actions possible par le pacman sinon return false
	 * @param listActions
	 * @param action
	 * @return
	 */
	public boolean isPossible(ArrayList<AgentAction> listActions, int action){
		boolean possible=false;
		int i=0;
		while(!possible && i<listActions.size()){
			if(listActions.get(i).get_idAction()==action)
				possible=true;
			++i;
		}
		return possible;
	}

	/**
	 * retourne l'action n si elle est legale sinon renvoie une action aléatoire parmis les actions legales
	 * @param state
	 * @param n
	 * @return
	 */
	public AgentAction randomAction(PacmanGame state, int n) {

		ArrayList<AgentAction> actionsPossible = state.getLegalPacmanActions();

		if (n == 0 && isPossible(actionsPossible, AgentAction.NORTH)) {
			return new AgentAction(AgentAction.NORTH);
		} else if (n == 1 && isPossible(actionsPossible, AgentAction.SOUTH)) {
			return new AgentAction(AgentAction.SOUTH);
		} else if (n == 2 && isPossible(actionsPossible, AgentAction.EAST)) {
			return new AgentAction(AgentAction.EAST);
		} else if (n == 3 && isPossible(actionsPossible, AgentAction.WEST)) {
			return new AgentAction(AgentAction.WEST);
		} else {
			int indexAleatoire = (int) (Math.random() * actionsPossible.size());
			return actionsPossible.get(indexAleatoire);
		}

	}

	@Override
	public AgentAction chooseAction(PacmanGame state) {

		int n = random.nextInt(4);
		ArrayList<AgentAction> actionsPossible=state.getLegalPacmanActions();

		AgentAction actionChoisie=actionsPossible.get(0);

		if(random.nextDouble() < this.current_epsilon) {
		
			actionChoisie=randomAction(state, n);
			return actionChoisie;
		}else{
			double maxQvalue = -9999;

			for (AgentAction action : actionsPossible) {
				
				double[] features = extractFeatures(state, action);
				double qValue = this.nn.predict(features)[0];
				
				if(qValue > maxQvalue) {
					
					maxQvalue = qValue;
					actionChoisie = action;
					
				} else if(qValue == maxQvalue) {
					return randomAction(state, n);
					
				}
			}

			return actionChoisie;
		}

	}


	@Override
	public void update(PacmanGame state, PacmanGame nextState, AgentAction action, double reward, boolean isFinalState) {

		double maxQvalue_nextState = -9999;

		if (!isFinalState) {

			for (int i = 0; i < this.d+1; i++) {
				
				AgentAction a = new AgentAction(i);
				if (nextState.isLegalMove(nextState.pacman, a)) {

					double[] features = extractFeatures(nextState, a);
					double nextStateQ = this.nn.predict(features)[0];

					if (nextStateQ > maxQvalue_nextState) {
						maxQvalue_nextState = nextStateQ;
					}
				}
			}

			double[] targetQ = new double[1];
			targetQ[0] = reward + gamma * maxQvalue_nextState;

			double[] features = extractFeatures(state, action);
			TrainExample trainExample = new TrainExample(features, targetQ);
			trainExamples.add(trainExample);
		}
	}


	@Override
	public void learn(ArrayList<TrainExample> trainExamples) {

		nn.fit(trainExamples, this.nEpochs, this.batchSize, this.learningRate);

	}

}

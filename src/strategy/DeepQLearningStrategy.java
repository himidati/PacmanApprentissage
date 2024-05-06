package strategy;

import java.util.ArrayList;
import java.util.List;

import agent.Agent;
import agent.AgentAction;

import agent.PositionAgent;
import agent.typeAgent;
import motor.Maze;
import motor.PacmanGame;
import neuralNetwork.NeuralNetWorkDL4J;

import neuralNetwork.TrainExample;

import java.util.Random;


public class DeepQLearningStrategy extends QLearningStrategy {

	int nEpochs;
	int batchSize;
	
	int range;
	int d=4;
	Random random= new Random();

	NeuralNetWorkDL4J nn;
	int sizeState;
	
	boolean modeAllMaze;
		
	
	public DeepQLearningStrategy(double epsilon, double gamma, double alpha, int range, int nEpochs, int batchSize,  int sizeMazeX, int sizeMazeY, boolean modeAllMaze, int nbWalls) {
		
		
		super(epsilon, gamma, alpha, sizeMazeX, sizeMazeY);

		
		this.modeAllMaze = modeAllMaze;
		
		System.out.println("nbWalls : " + nbWalls);
		
		if(modeAllMaze) {
			
			this.sizeState = (sizeMazeX)*(sizeMazeY)*4 - nbWalls;
			
		} else {
			this.sizeState = range*range*4;
		}
		
		System.out.println("Size entry neural network : " + this.sizeState);
		
		this.nn = new NeuralNetWorkDL4J(alpha, 0, sizeState, 4);
		
		this.nEpochs = nEpochs;
		this.batchSize = batchSize;
		
		this.range = range;
		
		
		
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
				posYPac=(posYPac+1)% this.sizeMazeY;;
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
	 * retourne l'action n si elle est possible sinon renvoie une action aléatoire parmis les actions legales
	 * @param state
	 * @param n
	 * @return
	 */
	public AgentAction randomAction(PacmanGame state, int n){

		ArrayList<AgentAction> actionsPossible=state.getLegalPacmanActions();
			
			if(n==0 && isPossible(actionsPossible, AgentAction.NORTH)){
				return new AgentAction(AgentAction.NORTH);
			}else if( n==1 && isPossible(actionsPossible, AgentAction.SOUTH)){
				return new AgentAction(AgentAction.SOUTH);
			}else if(n==2 && isPossible(actionsPossible, AgentAction.EAST)){
				return new AgentAction(AgentAction.EAST);
			}else if(n==3 && isPossible(actionsPossible, AgentAction.WEST)){
				return new AgentAction(AgentAction.WEST);
			}else{
				int indexAleatoire = (int) (Math.random() * actionsPossible.size());
				return actionsPossible.get(indexAleatoire);
			}

	}

	
	@Override
	public AgentAction chooseAction(PacmanGame state) {
		int n = random.nextInt(4);
		ArrayList<AgentAction> actionsPossible = state.getLegalPacmanActions();
		AgentAction actionChoisie = actionsPossible.get(0);

		if (random.nextDouble() < this.current_epsilon) {
			return randomAction(state, n);
		} else {
			double maxQvalue = -9999;

			for (AgentAction action : actionsPossible) {
				double[] encodeState = encodeState(state);
				double[] output = this.nn.predict(encodeState);

				if (output[action.get_idAction()] > maxQvalue) {
					maxQvalue = output[action.get_idAction()];
					actionChoisie = action;
				}
			}

			return actionChoisie;
		}
	}
	

	public double[] encodeState(PacmanGame state){	

		double[] encodeState=new double[this.sizeState];		

		int k=0;

		for(int i=0; i<this.sizeMazeX;i++){
			for(int j=0; j<this.sizeMazeY; j++){
				if (!state.isWallAtPosition(i, j)) {
					encodeState[k]=1;
					if (state.isCapsuleAtPosition(i, j)) {
						encodeState[k]=1;
					} else if (state.isPacmanAtPosition(i, j)) {
						encodeState[k]=1;
					} else if (state.isGhostAtPosition(i, j)) {
						encodeState[k]=1;
					} else if (state.isGumAtPosition(i, j)) {
						encodeState[k]=1;
					} else{
						encodeState[k]=0;
					}
				}else{
					encodeState[k]=0;
				}
				k++;
			}
						
		}
		return encodeState;
	}
	
	@Override
	public void update(PacmanGame state, PacmanGame nextState, AgentAction action, double reward,boolean isFinalState) {

		double maxQnextState = -9999;
		if (!isFinalState) {
			for (int i = 0; i < this.sizeState; i++) {
				AgentAction a = new AgentAction(i);
				if (nextState.isLegalMove(nextState.pacman, a)) {
					double[] encodedState = encodeState(nextState);
					double[] nextStateQ = this.nn.predict(encodedState);

					for (double v : nextStateQ) {
						if (v > maxQnextState) {
							maxQnextState = v;
						}
					}
				}
			}

			double[] encodedState = encodeState(state);
			double[] targetQ = this.nn.predict(encodedState);
			targetQ[action.get_idAction()] = reward + gamma * maxQnextState;

			TrainExample trainExample = new TrainExample(encodedState, targetQ);
			trainExamples.add(trainExample);
		}
	}

	
	
	public void learn(ArrayList<TrainExample> trainExamples) {
		
		nn.fit(trainExamples, nEpochs, batchSize, learningRate);
	}
	
	
}

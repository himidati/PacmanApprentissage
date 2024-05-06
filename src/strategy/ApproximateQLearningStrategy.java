package strategy;

import java.util.ArrayList;
import java.util.List;

import agent.Agent;
import agent.AgentAction;
import agent.PositionAgent;
import motor.Maze;
import motor.PacmanGame;
import neuralNetwork.TrainExample;

import java.util.Random;


public class ApproximateQLearningStrategy extends QLearningStrategy{

	double[] weights;
	int d=3;

	double[] current_f;
	Random random= new Random();
	
	public ApproximateQLearningStrategy(double epsilon, double gamma, double alpha, int sizeMazeX, int sizeMazeY) {
		super(epsilon, gamma, alpha, sizeMazeX, sizeMazeY);
		
		this.weights=new double[d+1];
		
		for(int i=0; i<=d; i++){
			this.weights[i]=random.nextGaussian();
		}
	}

	/**
	 * renvoie vrai si des fantômes se trouvent dans une case au tour du pacman
	 * après avoir joué l'action action à l'état state (on suppose que les fantômes ne bougent pas entre eux)
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

	public Boolean isWall(PacmanGame state, AgentAction action) {
		int x=state.getPacmanX();
		int y=state.getPacmanY();

		switch (action.get_idAction()) {
			case 0:
				if (y > 0) y -= 1;  else y=this.sizeMazeY-1;
				return (state.isWallAtPosition(x, y)) ;
			case 1:
				y=(y+1) % this.sizeMazeY;;
				return (state.isWallAtPosition(x, y));
			case 2:
				x=(x+1)% this.sizeMazeX;
				if(state.isWallAtPosition(x, y)) ;
			case 3:
				if (x > 0) x -= 1; else x=this.sizeMazeX-1;
				return (state.isWallAtPosition(x, y));
		}
		return false;
	}

	public double[] extractFeatures(PacmanGame state, AgentAction action ){
		double[] f =new double[d+1];

		f[0]=1;
		f[1]=canEat(state, action) ? 12.0 : 0.0;
		f[2]=(nbCoup(state, action)<3 && nbCoup(state, action)!=0)? 10 : 0.0;
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


	public double scalarProduct(double[] w, double[] f) {

		double q = 0;

		for(int i = 0; i < w.length; i++) {

			q += w[i]*f[i];
		}

		return q;
	}

	@Override
	public AgentAction chooseAction(PacmanGame state) {
		double[] fn = extractFeatures(state, new AgentAction(AgentAction.NORTH));
		double qn = scalarProduct(this.weights,  fn);
	
		double[] fs = extractFeatures(state, new AgentAction(AgentAction.SOUTH));
		double qs = scalarProduct(this.weights,  fs);

		double[] fe = extractFeatures(state, new AgentAction(AgentAction.EAST));
		double qe = scalarProduct(this.weights, fe);

		double[] fw = extractFeatures(state, new AgentAction(AgentAction.WEST));
		double qw = scalarProduct(this.weights,  fw);

		ArrayList<AgentAction> actionsPossible=state.getLegalPacmanActions();
		
		if(random.nextDouble() < this.current_epsilon) {
			int n = random.nextInt(4);
			if(n==0 && isPossible(actionsPossible, AgentAction.NORTH)){
				this.current_f = fn;
				return new AgentAction(AgentAction.NORTH);
			}else if( n==1 && isPossible(actionsPossible, AgentAction.SOUTH)){
				this.current_f=fs;
				return new AgentAction(AgentAction.SOUTH);
			}else if(n==2 && isPossible(actionsPossible, AgentAction.EAST)){
				this.current_f=fe;
				return new AgentAction(AgentAction.EAST);
			}else if(n==3 && isPossible(actionsPossible, AgentAction.WEST)){
				this.current_f=fw;
				return new AgentAction(AgentAction.WEST);
			}else{
				
				int indexAleatoire = (int) (Math.random() * actionsPossible.size());
				double[] fa = extractFeatures(state,actionsPossible.get(indexAleatoire));
				this.current_f=fa;
				return actionsPossible.get(indexAleatoire);
			}
		}else{

			double maxQ = Math.max(Math.max(qn, qs), Math.max(qe, qw));
			if( maxQ == qn && isPossible(actionsPossible, AgentAction.NORTH) && qn!=qs) {
				this.current_f = fn;
				return  new AgentAction(AgentAction.NORTH);

			}else if(maxQ == qs && isPossible(actionsPossible, AgentAction.SOUTH) && qs!=qn ) {
				this.current_f = fs;
				return new AgentAction(AgentAction.SOUTH);

			}else if(maxQ == qe && isPossible(actionsPossible, AgentAction.EAST) && qe!=qw) {
				this.current_f = fe;
				return new AgentAction(AgentAction.EAST);

			}else if(maxQ == qw && isPossible(actionsPossible, AgentAction.WEST ) && qw!=qe) {
				this.current_f = fw;
				return new AgentAction(AgentAction.WEST);
			}
			else  { //action aléatoire parmis les actions possibles
				
				int indexAleatoire = (int) (Math.random() * actionsPossible.size());
				double[] fa = extractFeatures(state,actionsPossible.get(indexAleatoire));
				this.current_f=fa;
				return actionsPossible.get(indexAleatoire);
			}
		}
	}
	
	
	@Override
	public void update(PacmanGame state, PacmanGame nextState, AgentAction action, double reward,boolean isFinalState) {

		double[] fn = extractFeatures(nextState, new AgentAction(AgentAction.NORTH));
		double qn = scalarProduct(this.weights, fn);

		double[] fs = extractFeatures(nextState, new AgentAction(AgentAction.SOUTH));
		double qs = scalarProduct(this.weights, fs);

		double[] fe = extractFeatures(nextState, new AgentAction(AgentAction.EAST));
		double qe = scalarProduct(this.weights, fe);

		double[] fw = extractFeatures(nextState, new AgentAction(AgentAction.WEST));
		double qw = scalarProduct(this.weights, fw);

		double target = reward + this.gamma * Math.max(Math.max(qn, qs), Math.max(qe, qw));
		double Qstate = scalarProduct(this.weights, this.current_f);

		for (int i = 0; i <= d; i++) {
			this.weights[i] = this.weights[i] - 2 * this.learningRate * this.current_f[i] * (Qstate - target);
		}
		
	}

	
	@Override
	public void learn(ArrayList<TrainExample> trainExamples) {
	
	}
	
	
	
	
	
	
	

	
	

}

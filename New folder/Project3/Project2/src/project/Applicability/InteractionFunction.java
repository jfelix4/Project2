package project.Applicability;

import project.Interaction.*;
import project.Structure.*;


public class InteractionFunction {

	public static WelcomeView WelcomeView = new WelcomeView();
    public static FileHandlingChoices FileHandlingChoices = new FileHandlingChoices();
    
    

    public static Interaction ActualView = WelcomeView;

    
    public static Interaction getActualView() {
        return ActualView;
    }

    
    public static void setActualView(Interaction actualView) {
    	ActualView = actualView;
    }
    
    
}

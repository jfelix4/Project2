package project.Interaction;

import project.Applicability.*;
import java.util.InputMismatchException;
import java.util.Scanner;
import java.util.ArrayList;


public class WelcomeView implements Interaction {

	private String welcomeText     = "Welcome to Lockers Pvt. Ltd!";
    private String developerText   = "Developer: Jhonny Felix";
    private String developerEmail  = "Emaail: jfelix@gmail.com";
    private String productDetails  = "Name: Digitalization, Lockedme.com";
    private String releaseDate     = "Release: January 2022";
    private ArrayList<String> options = new ArrayList<>();


    public WelcomeView() {
        options.add("1. Display Files");
        options.add("2. Display File Choices Menu");
        options.add("3. Leave The Program");

    }
    
    public void introWS() {
    	System.out.println(welcomeText);
        System.out.println(developerText);
        System.out.println(developerEmail);
        System.out.println(productDetails);
        System.out.println(releaseDate);
        System.out.println("-------------------------------------------");
        Display();
    }
    
    
    
    @Override
    public void Display() {
    	System.out.println("Main Menu");
        for (String s : options)  {
            System.out.println(s);
        }

    }

    public void GetUserInput() {
        int selectedOption  = 0;
        while ((selectedOption = this.getOption()) != 3) {
            this.NavigateOption(selectedOption);
        }
    }

    @Override
    public void NavigateOption(int option) {
        switch(option) {

            case 1: // Show Files in Directory
                this.ShowFiles();
                
                this.Display();
                
                break;
                
            case 2: // Show File Options menu
            	InteractionFunction.setActualView(InteractionFunction.FileHandlingChoices);
            	InteractionFunction.getActualView().Display();
            	InteractionFunction.getActualView().GetUserInput();
                
                this.Display();
                
                break;
                
            default:
                System.out.println("Invalid Option");
                break;
        }
        
    }

    public void ShowFiles() {

        //files from the Directory
    	

        System.out.println("List of Files: ");
    	DirectorySolution.PrintFiles();

    }
    
    private int getOption() {
        Scanner in = new Scanner(System.in);

        int returnOption = 0;
        try {
            returnOption = in.nextInt();
        }
        catch (InputMismatchException ex) {

        }
        return returnOption;

    }
}

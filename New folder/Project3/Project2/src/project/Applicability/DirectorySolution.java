package project.Applicability;

import project.Structure.*;
import java.io.File;


public class DirectorySolution {
    private static Directory fileDirectory = new Directory();
    
    public static void PrintFiles() {
    	
    	fileDirectory.chargeFiles();

        for (File file : DirectorySolution.getFileDirectory().getFiles())
        {
            System.out.println(file.getName());
        }
    }
    public static Directory getFileDirectory() {
        return fileDirectory;
    }

    public static void setFileDirectory(Directory fileDirectory) {
        DirectorySolution.fileDirectory = fileDirectory;
    }


}



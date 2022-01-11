package project.Structure;

import java.util.ArrayList;
import java.util.Collections;
import java.io.File;
import java.nio.file.FileSystems;
import java.nio.file.Path;


public class Directory {

   public static final String name = "Project/src/project/Structure/Directory";

    private ArrayList<File> files = new ArrayList<File>();
    
    Path path = FileSystems.getDefault().getPath(name).toAbsolutePath();
    
    File Afiles = path.toFile();
       
    public String getName() {
        return name;
    }
    
    public void print() {
    	System.out.println("Files in Directory: ");
    	files.forEach(f -> System.out.println(f));
    }

    public ArrayList<File> chargeFiles() {
    	
        File[] directoryFiles = Afiles.listFiles();
        
        
        
    	files.clear();
    	for (int i = 0; i < directoryFiles.length; i++) {
    		if (directoryFiles[i].isFile()) {
    			files.add(directoryFiles[i]);
    		}
    	}
    	
    	Collections.sort(files);
    	
    	return files;
    }

    public ArrayList<File> getFiles() {
    	
    	chargeFiles();
    	return files;
    }
    
    
    public void mergeSort() {
    	
    }
    
}

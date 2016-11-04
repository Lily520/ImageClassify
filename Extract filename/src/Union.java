import java.io.BufferedWriter;
import java.io.File;
import java.io.FileWriter;
import java.io.IOException;
//列出所有视频的名称列表到一个文件中

public class Union {
    public static void main(String[] args) {
        String path="C:\\Users\\Administrator\\Desktop\\毕设\\datasets\\caltech\\101_ObjectCategories\\ant";
        String path2="C:\\Users\\Administrator\\Desktop\\毕设\\datasets\\caltech\\101_ObjectCategories\\ant.txt";
        File file=new File(path);
        //File[] tempList = file.listFiles();
        String [] tempList = file.list();
        System.out.println("该目录下对象个数："+tempList.length);
        for (int i = 0; i < tempList.length; i++) {
            // if (tempList[i].isFile()) {
            //System.out.println("文 件："+tempList[i]);
            Union.writeToFile(path2,tempList[i].toString()+"\r\n");

            // }
            //if (tempList[i].isDirectory()) {
            //  System.out.println("文件夹："+tempList[i]);
            //}

        }
        //Union.writeToFile(path2,tempList.toString());
    }
    public static void writeToFile(String filename,String content){

        try {
            BufferedWriter out = new BufferedWriter(new FileWriter(filename, true));
            out.write(content);
            out.close();
        }
        catch (IOException e) { System.out.println("Error in writing to output file!"); }

        // System.out.println("outliers: " + nonmembers);
    }
}
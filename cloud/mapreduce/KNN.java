import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStreamReader;
import java.net.URI;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.Iterator;
import java.util.Set;
import org.apache.hadoop.util.GenericOptionsParser;
import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.FSDataInputStream;
import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.IOUtils;
import org.apache.hadoop.io.LongWritable;
import org.apache.hadoop.io.NullWritable;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.mapreduce.Job;
import org.apache.hadoop.mapreduce.Mapper;
import org.apache.hadoop.mapreduce.Reducer;
import org.apache.hadoop.mapreduce.lib.input.FileInputFormat;
import org.apache.hadoop.mapreduce.lib.input.TextInputFormat;
import org.apache.hadoop.mapreduce.lib.output.FileOutputFormat;
import org.apache.hadoop.mapreduce.lib.output.TextOutputFormat;
import org.apache.hadoop.mapreduce.lib.partition.HashPartitioner;
import java.util.*;

//an instance,include <key(int),value(a1,a2,...ak> value is a list 
class Element
{
    private int key;//storage key
    private double[] attributeset;//storage value
    public  Element(String str)
    {
        String[] splited = str.split(" ");
        attributeset = new double[splited.length-1];
        for(int i=0;i<attributeset.length;i++)
        {
            attributeset[i] = Double.parseDouble(splited[i+1]);  
        }
        key = Integer.parseInt(splited[0]);      
    }
    public double[] getAttributeset()
    {
        return attributeset; 
    }
    public int getKey()
    {
        return key;
    }
}

class Distance
{
    public static double EuclideanDistance(double[] a,double[] b)
    {
        double sum = 0.0;
        for(int i=0;i<a.length;i++)
        {
            sum += Math.pow(a[i]-b[i],2);   
        }    
        return Math.sqrt(sum);
    }
}


public class KNN
{   
    public static String path1;//read train data
    public static String path2;//output data


    public static void main(String[] args) throws Exception
    {
        /*
        Element_test t = new Element_test("1 2.0 3.0 4.0 5.0 6.0");
        System.out.println(t.getKey());
        t.print();
        */
        
        // 组件配置
        Configuration conf = new Configuration();
        //get args data
        String[] myArgs = new GenericOptionsParser(conf, args).getRemainingArgs();
        if (myArgs.length != 2) {
            System.err.println("Usage: KNN <in> <out>");
            System.exit(2);
        }
        path1 = myArgs[0];// MapReduce的输入文件夹
        path2 = myArgs[1];// MapReduce的输出文件夹
        //if the file had exite, then delete it
        FileSystem fileSystem = FileSystem.get(new Configuration());
        if(fileSystem.exists(new Path(path2)))
        {
            fileSystem.delete(new Path(path2), true);
        }
        //new job, and its' job is KNN
        Job job = new Job(new Configuration(),"KNN");
        job.setJarByClass(KNN.class);
        FileInputFormat.setInputPaths(job, new Path(path1));//set input path
        job.setInputFormatClass(TextInputFormat.class);//set format
        job.setMapperClass(MyMapper.class);//mapper
        job.setMapOutputKeyClass(Text.class);//output format
        job.setMapOutputValueClass(Text.class);//output format

        job.setPartitionerClass(HashPartitioner.class);

        job.setReducerClass(MyReducer.class);
        job.setOutputKeyClass(Text.class);
        job.setOutputValueClass(NullWritable.class);
        job.setOutputFormatClass(TextOutputFormat.class);
        //set output path
        FileOutputFormat.setOutputPath(job, new Path(path2));
        job.waitForCompletion(true);
        

    }
    public static class MyMapper extends Mapper<LongWritable, Text, Text, Text>
    {
        //trainset to storge all elements
        public ArrayList<Element> trainSet = new ArrayList<Element>();
        public int k = 4;//KNN->k

        //read data and begin to work
        protected void setup(Context context)throws IOException, InterruptedException
        {
            FileSystem fileSystem = null;  
            try  
            {  
                Configuration config = new Configuration();
                config.set("fs.default.name", "hdfs://master:9000");
                URI uri=new URI("hdfs://master:9000");
                fileSystem = FileSystem.get(uri,config);  
            } catch (Exception e){}  

            //read data from hdfs://master:9000/input1/mnist0
            FSDataInputStream fs = fileSystem.open(new Path("hdfs://master:9000/input1/test_max"));   
            BufferedReader train_flow = new BufferedReader(new InputStreamReader(fs));   
            // get a trainData
            String str = train_flow.readLine();  
            //String str = "1 2.0 3.0 4.0 5.0 6.0";
            while(str!=null)  
            {  
                Element trainElement = new Element(str);
                trainSet.add(trainElement);
                str = train_flow.readLine();  
            }
        }
        
        protected void map(LongWritable k1, Text v1,Context context)throws IOException, InterruptedException
        {
            /* 
            ** storage TreeHash, 
            ** Automatically ordered,
            ** the method is give by function compare
            */
            Map<Double, String> mTreeMap = new TreeMap<Double, String>(new Comparator<Double>(){
                public int compare(Double o1, Double o2) {
                    return o1.compareTo(o2);
                }
            });

            /* 
            ** get a test element
            ** this element is read in hdfs://master:9000/input/data, call test data
            ** calculate the diatance from the testData to every trainData and insert into TreeHash 
            */
            Element testElement = new Element(v1.toString());
            for(int i=0;i<trainSet.size();i++)
            {
                //calculate the diatance
                double dis = Distance.EuclideanDistance(trainSet.get(i).getAttributeset(),testElement.getAttributeset());
                //put the element into the TreeHash, include its' distance
                mTreeMap.put(dis, trainSet.get(i).getKey()+"");
            }
            
            /*
            ** deal with the first k TreeHash,call KNN
            ** pass the first k data to reduce
            ** the form is <key, value>
            ** we can see in context 
            */
            int cnt = 0;
            Iterator<Map.Entry<Double, String>> iter = mTreeMap.entrySet().iterator();
            String t1 = "", t2 = "";
            while(iter.hasNext())
            {
                Map.Entry<Double, String> map = iter.next();
                cnt++;
                t1 += Double.toString(map.getKey());
                t2 += map.getValue();
                if(cnt != k) {
                    t1 += " ";
                    t2 += " ";
                }
                //context.write(new Text(Double.toString(map.getKey())), new Text(map.getValue()));
                if(cnt == k) {
                    break;
                }
            }
            t2 += " ";
            t2 += Integer.toString(testElement.getKey());
            context.write(new Text(t1), new Text(t2));
            
        }
    }

    public static class MyReducer  extends Reducer<Text, Text, Text, NullWritable>
    {   
        
        //k2 is the distance v2s is the <key,value>
        protected void reduce(Text k2, Iterable<Text> v2s,Context context)throws IOException, InterruptedException
        {
            /*
            ** process the data passed from map
            ** reduce the data to a array 
            ** and every has key and value
            */
            HashMap<String, Double> temp = new HashMap<String, Double>(); 
            ArrayList<String> arr = new ArrayList<String>();
            String arr_str = "";
            for (Text v2 : v2s)
            { 
                arr_str += v2.toString();  
            }
            String pre1 = "", pre2 = "";
            for(int i = 0; i < (k2.toString()).length(); i++) {
                if((k2.toString()).charAt(i) != '['||(k2.toString()).charAt(i) != ']') {
                    pre1 += (k2.toString()).charAt(i);
                }
            }
            for(int i = 0; i < arr_str.length(); i++) {
                if(arr_str.charAt(i) != '['||arr_str.charAt(i) != ']') {
                    pre2 += arr_str.charAt(i);
                }
            }
            String[] splited_dis = (pre1).split(" ");
            String[] splited_id = (pre2).split(" ");

            /*
            for(int i =0; i < splited_id.length - 1; i++)
                context.write(new Text(splited_id[i]+" "+splited_dis[i]),NullWritable.get());
            */

            /*
            ** put the <key, value> data into temp(HashMap)
            ** key is id, value is the count in this k data pair
            */
            String testId = splited_id[splited_id.length-1];
            for(int i = 0; i < splited_id.length - 1; i++) {
                if(!temp.containsKey(splited_id[i])) {
                    temp.put(splited_id[i], new Double(1));
                } else {
                    double frequence = temp.get(splited_id[i])+1;
                    temp.remove(splited_id[i]);
                    temp.put(splited_id[i],frequence);
                }
            }

            /*
            ** fin the largest count in <id, count> pair
            ** which means that id appears most in this first k data pair
            ** and this id indecates that it's the id we want to predect
            */
            Set<String> s = temp.keySet();
            Iterator it = s.iterator();
            double max=Double.MIN_VALUE;
            String predictlable = "";
            while(it.hasNext()) {
                String key = (String)it.next();
                Double count = temp.get(key);
                if(count > max)
                {
                    max = count;
                    predictlable = key;
                }
            }
            context.write(new Text("test id: " + testId + "   predict group : " + predictlable),NullWritable.get());
        } 
    }
}







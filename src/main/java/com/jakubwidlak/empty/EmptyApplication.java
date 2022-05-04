package com.jakubwidlak.empty;

import org.springframework.boot.SpringApplication;
import org.springframework.boot.autoconfigure.SpringBootApplication;

import java.util.*;

@SpringBootApplication
public class EmptyApplication {

	public static void main(String[] args) {


		one();


		//SpringApplication.run(EmptyApplication.class, args);
	}

	public static void quicksort(Comparable[] a)
	{
		Collections.shuffle(Arrays.asList(a));
		sort(a, 0, a.length - 1);
	}
	private static void sort(Comparable[] a, int lo, int hi)
	{
		if (hi <= lo) return;
		int j = partition(a, lo, hi);
		sort(a, lo, j-1);
		sort(a, j+1, hi);
	}

	private static int partition(Comparable[] a, int lo, int hi)
	{
		int i = lo, j = hi+1;
		while (true)
		{
			while (a[++i].compareTo(a[lo])<0)
				if (i == hi) break;
			while (a[lo].compareTo(a[--j])<0)
				if (j == lo) break;
			if (i >= j) break;
			exch(a, i, j);
		}
		exch(a, lo, j);
		return j;
	}

	private static void exch(Comparable[] a, int i, int j)
	{
		Comparable swap = a[i];
		a[i] = a[j];
		a[j] = swap;
	}

	public static void bottomUpMergesort(Comparable[] a){
		int N = a.length;
		Comparable[] aux = new Comparable[N];
		for(int size=1; size<N;size+=size){
			for (int lo=0;lo<N-size; lo +=size+size){
				merge(a, aux, lo, lo+size-1, Math.min(lo+size+size-1, N-1));
			}
		}
	}

	public static void mergesort(Comparable[] a)
	{
		Comparable[] aux = new Comparable[a.length];
		sort(a, aux, 0, a.length - 1);
	}

	private static void sort(Comparable[] a, Comparable[] aux, int lo, int hi)
	{
		if (hi <= lo) return;
		int mid = lo + (hi - lo) / 2;
		sort(a, aux, lo, mid);
		sort(a, aux, mid+1, hi);
		if (!(a[mid+1].compareTo(a[mid])<0)) return;
		merge(a, aux, lo, mid, hi);
	}

	private static void merge(Comparable[] a, Comparable[] aux, int lo, int mid, int hi)
	{
		for (int k = lo; k <= hi; k++)
			aux[k] = a[k];
		int i = lo, j = mid+1;
		for (int k = lo; k <= hi; k++)
		{
			if (i > mid) a[k] = aux[j++];
			else if (j > hi) a[k] = aux[i++];
			else if (aux[j].compareTo(aux[i])<0) a[k] = aux[j++];
			else a[k] = aux[i++];
		}
	}

	static void LSD(){
		String[] input = {"1234", "4321", "2314", "4432", "1233", "2134"};
		String[]a = input;
		int W=4;
		int R = 256;
		int N = a.length;
		String[] aux = new String[N];
		for (int d = W-1; d >= 0; d--)
		{
			int[] count = new int[R+1];
			for (int i = 0; i < N; i++)
				count[a[i].charAt(d) + 1]++;
			for (int r = 0; r < R; r++)
				count[r+1] += count[r];
			for (int i = 0; i < N; i++)
				aux[count[a[i].charAt(d)]++] = a[i];
			for (int i = 0; i < N; i++)
				a[i] = aux[i];
		}
		System.out.println(Arrays.toString(a));

		for(int i= input[0].length()-1; i >=0; i--){
			int[] count = new int[5];
			for (int j=0; j< input.length; j++){
				int digit = input[j].charAt(i) - '0';
				count[digit]++;
			}
			for (int j=0; j<4; j++){
				count[j+1]+=count[j];
			}
			for(int j = 0; j<input.length; j++){
				int digit = input[j].charAt(i)-'0';
				int temp = count[digit-1]++;
				aux[temp]= input[j];
			}
			for (int j=0; j<count.length; j++){
				input[j]=aux[j];
			}
		}
		System.out.println(Arrays.toString(input));
	}

	static void graphOne(){
		Graph g = new Graph();
		g.addEdge(0,1, 3);
		g.addEdge(1,2, 1);
		g.addEdge(1,3, 4);
		g.addEdge(2,3, 1);

		g.print();
	}

	int findNodeCount(Graph g, int node){
		int count=0;
		int threshold=4;
		int distTo[] = new int[g.nodes.size()];
		int edgeTo[] = new int[g.nodes.size()];
		PriorityQueue<GNode> q= new PriorityQueue<>();
		q.add(g.nodes.get(node));
		while(!q.isEmpty()){

		}

		return count;
	}

	void relax(Edge e, int[] distTo, Edge[] edgeTo){
		int v = e.either().label, w = e.other(e.either()).label;
		if(distTo[w] > distTo[v]+e.weight){
			distTo[w] = distTo[v] +e.weight;
			edgeTo[w] = e;
		}
	}

	static class Graph{
		HashMap<Integer, GNode> nodes = new HashMap<>();

		void addEdge(int u, int v, int weight){
			GNode un = getNode(u), vn = getNode(v);
			Edge edge=new Edge(un, vn, weight);
			un.edges.add(edge);
			vn.edges.add(edge);
		}

		GNode getNode(int label){
			GNode result;
			if(nodes.containsKey(label)){
				result = nodes.get(label);
			}else{
				result = new GNode();
				result.label=label;
				nodes.put(label, result);
			}
			return result;
		}

		void print(){
			for (GNode node: nodes.values()){
				System.out.print(node.label + " -> [");
				for(Edge edge: node.edges){
					System.out.print(edge.other(node).label + " ");
				}
				System.out.println("]");
			}
		}
	}

	static class Edge {

		GNode u, v;
		int weight;

		Edge(GNode u, GNode v, int weight){
			this.u = u;
			this.v=v;
			this.weight=weight;
		}

		GNode either(){return u;}

		GNode other(GNode x){
			if (x==u) return v;
			return u;
		}
	}

	static class GNode{
		int label;
		HashSet<Edge> edges = new HashSet<>();
	}

	static void eighteen(){
		//Reverse a linked list
		Node<Integer> root = new Node<>(null, 5, null);
		root = new Node<>(null, 4, root);
		root = new Node<>(null, 3, root);
		root = new Node<>(null, 2, root);
		root = new Node<>(null, 1, root);

		Node temp = root;
		while (temp!=null){
			System.out.print(temp.item);
			temp=temp.next;
		}
		System.out.println();
		root=reverse(root);
		while (root!=null){
			System.out.print(root.item);
			root=root.next;
		}

	}

	static Node reverse(Node head){
		Node reversed = head;
		Node temp;

		temp = head;
		head = head.next;
		temp.next=null;
		reversed=temp;

		while (head!=null){
			temp = head;
			head = head.next;
			temp.next=reversed;
			reversed=temp;
		}

		return reversed;
	}

	static void seventeen(){
		//Find the missing number
		int[] in = {8, 3, 5, 2, 4, 6, 0, 1};
		int sum=0, expected=0;
		for (int j : in) {
			sum += j;
		}
		expected=in.length*(in.length+1)/2;
		System.out.println(expected-sum);

		Integer[] input = {8, 3, 5, 2, 4, 6, 0, 1};
		TreeSet<Integer> set = new TreeSet<>();
		Collections.addAll(set, input);
		for(int i=0; i<input.length; i++){
			if(!set.contains(i)){
				System.out.println(i);
				return;
			}
		}
	}

	static void sixteen(){
		//Path sum
		int value = 26;
		BinaryTree tree = new BinaryTree();
		tree.add(5);
		tree.add(4);
		tree.add(8);
		tree.add(11);
		tree.add(13);
		tree.add(0);
		tree.add(4);
		tree.add(7);
		tree.add(0);
		tree.add(0);
		tree.add(0);
		tree.add(2);
		tree.add(0);
		tree.add(0);
		tree.add(1);
		tree.print();

		System.out.println(pathSum(tree.root, value));
	}

	static boolean pathSum(IntNode node, int val){
		if (node==null) return false;
		if (node.item==val) return true;
		val-=node.item;
		boolean result = false;
		//if (node.left.item<=val){
			result = pathSum(node.left, val);
		//}
		//if (node.right.item<=val){
			result |= pathSum(node.right, val);
		//}
		return result;
	}

	static boolean isPowerOfTwo(int x)
	{
        /* First x in the below expression is
        for the case when x is 0 */
		return x != 0 && ((x & (x - 1)) == 0);
	}

	static void fifteen(){
		//Merge overlapping intervals
		int[][] input = {{1,3}, {2,9}, {8,10}, {15,18}};

		for(int i=0; i<input.length-1;i++){
			if(input[i][1]>input[i+1][0]){
				input[i+1]=new int[]{input[i][0], input[i+1][1]};
				input[i]=new int[0];
			}
		}
		for (int i=0; i<input.length;i++){
			if(input[i].length>0){
				System.out.print("["+input[i][0]+" "+input[i][1]+"]");
			}
		}
	}

	static void fourteen(){
		//Find Low/High Index
		int[] input = {3, 5, 6, 7, 7, 8, 8, 10, 15};
		int target =1;
		search(input, target, 0, input.length);
	}

	static int[] search(int[] arr, int target, int start, int end){
		int index=(end+start)/2;
		int val = arr[index];
		if (val==target){
			end=start=index;
			while(--start>0 && arr[start]==target){}
			while (++end<arr.length && arr[end]==target){}
			start++; end--;
			System.out.println(start+ " "+ end);
			return new int[]{start, end};

		}
		if(start==end){
			return new int[]{-1, -1};
		}
		if(val>target){
			return search(arr, target, start, index-1);
		}else{
			return search(arr, target, index+1, end);
		}
	}

	static void thirteen(){
		//LRU
		Cache cache = new Cache();
		cache.put(1, 1);
		cache.put(2, 2);
		cache.get(1);       // returns 1
		cache.put(3, 3);    // evicts key 2
		cache.get(2);       // returns -1 (not found)
		cache.put(4, 4);    // evicts key 1
		cache.get(1);       // returns -1 (not found)
		cache.get(3);       // returns 3
		cache.get(4);       // returns 4
	}

	static class Cache{
		int capacity =2;
		LinkedHashMap<Integer, Integer> cache = new LinkedHashMap<Integer, Integer>(16, 0.75F,true){
			@Override
			protected boolean removeEldestEntry(Map.Entry<Integer, Integer> eldest){
				return size()>capacity;
			}
		};

		void put(int k, int v){
			cache.put(k, v);
		}

		void get(int k){
			System.out.println(cache.get(k));
		}
	}

	static void twelve(){
		//Print balanced brace combinations
		int n = 4;
		char[] str = new char[2 * n];
		tw(str, 0,n, 0, 0);
	}

	static void tw(char str[], int index, int n, int left, int right){
		if(right==n){
			for (int i = 0; i < str.length; i++)
				System.out.print(str[i]);
			System.out.println();
			return;
		}
		if(left<n){
			str[index]='(';
			tw(str, index+1, n, left+1, right);
		}
		if (right<left){
			str[index]=')';
			tw(str, index+1,n, left, right+1);
		}
	}

	static void eleven(){
		//Determine if the number is valid
		Character[] help = {'1', '2', '3', '4', '5', '6', '7', '8', '9','0', '.'};
		HashSet<Character> dict = new HashSet<>();
		Collections.addAll(dict, help);
		boolean dotFound=false;
		String input = "22.22.";
		if(input.charAt(0)=='.'||input.charAt(input.length()-1)=='.'){
			System.out.println("Invalid");
			return;
		}
		for(int i=0; i<input.length();i++){
			if(dict.contains(input.charAt(i))){
				if(input.charAt(i)=='.'){
					if (dotFound) {
						System.out.println("Invalid");
						return;
					} else {
						dotFound=true;
					}
				}
			}else{
				System.out.println("Invalid");
				return;
			}
		}
		System.out.println("Valid");
	}

	static void ten(){
		//Largest sum subarray
		//int[] arr = {-2, 1, 2, 1, -3, 4, -1, 2, 1, -5, 4};
		//int[] arr = {-2, 1,2,1 };
		int[] arr = {-2, -1,-2,-1, -3 };
		int globalMax, currMax=globalMax=arr[0];
		int start, s, end = start = s= 0;
		for(int i=0; i < arr.length; i++){
			if(currMax<0){
				currMax=arr[i];
				s=i;
			}else{
				currMax+=arr[i];
			}
			if (globalMax<currMax) {
				globalMax = currMax;
				end=i;
				start=s;
			}
		}
		System.out.println(start + " " + end);
		System.out.println(globalMax);
	}

	static void nine(){
		//Find all palindrome substrings
		String input = "bananakayaktenet";
		int result = 0;
		for (int i = 1; i < input.length()-1; i++){
			int index=1;
			while (true){
				if(i-index<0 || i + index > input.length()-1) break;
				if (input.charAt(i-index)==input.charAt(i+index)){
					result++;
					index++;
				} else{
					break;
				}

			}

		}
		System.out.println(result);
	}

	static void eight(){
		//String segmentation
		String input = "appleppenapples";
		String[] dict = {"apple", "s", "pen", "app", "lep"};
		TreeSet<String> set = new TreeSet<>(Comparator.reverseOrder());
		Collections.addAll(set, dict);
		String temp = input;
		while (temp.length()>0){
			int start = temp.length();
			for (String sub: set){
				if (temp.startsWith(sub)){
					temp = temp.substring(sub.length());
				}
			}
			if(start==temp.length()) {
				System.out.println("false");
				return;
			}
		}
		System.out.println("true");
	}

	static boolean eighttwo(){
		String input = "applepenapple";
		String[] dict = {"apple", "pen"};
		HashSet<String> set = new HashSet<>();
		Collections.addAll(set, dict);
		return eighttwotemp(input,set);
	}

	static boolean eighttwotemp(String input, HashSet<String> set) {
		//String segmentation
		for(int i=0; i< input.length();i++){
			String first = input.substring(0, i);
			String second = input.substring(i);
			if (set.contains(first)){
				if (set.contains(second) || second.length()==0){
					return true;
				} else{
					return eighttwotemp(second, set);
				}
			}
		}
		return false;
	}

	static void seven(){
		//Check if two binary trees are identical
		BinaryTree tree = new BinaryTree();
		tree.add(40);
		tree.add(100);
		tree.add(400);
		tree.add(150);
		tree.add(50);
		tree.add(300);
		tree.add(600);

		BinaryTree tree2 = new BinaryTree();
		tree2.add(40);
		tree2.add(100);
		tree2.add(400);
		tree2.add(150);
		tree2.add(50);
		tree2.add(300);

		var result =tree.identicalTo(tree.root, tree2.root);
		System.out.println(result);
	}

	static void six(){
		//Mirror binary tree nodes
		BinaryTree tree = new BinaryTree();
		tree.add(40);
		tree.add(100);
		tree.add(400);
		tree.add(150);
		tree.add(50);
		tree.add(300);
		tree.add(600);

		tree.print();
		tree.reverse();
		System.out.println();
		tree.print();
	}

	private static class BinaryTree {
		IntNode root;

		boolean identicalTo(IntNode root1, IntNode root2){
			if(root1!=root2) return false;
			if(root1==null) return true;
			boolean left = identicalTo(root1.left, root2.left);
			boolean right = identicalTo(root1.right, root2.right);
			return left&&right;
		}

		void add(Integer val){
			root = add(root, val);
		}

		void reverse(){
			reverse(root);
		}

		void reverse(IntNode node){
			if(node==null) return;
			IntNode temp = node.left;
			node.left=node.right;
			node.right=temp;
			reverse(node.left);
			reverse(node.right);
		}

		void print(){
			Queue<IntNode> q = new LinkedList<>();
			q.add(root);
			int level=1;
			while (!q.isEmpty()){
				print(q);
				level++;
				if (isPowerOfTwo(level)){
					System.out.println();
				}
			}

		}

		void print (Queue<IntNode> q){
			IntNode node = q.poll();
			if (null!=node.left) q.add(node.left);
			if (null!=node.right) q.add(node.right);
			System.out.print(" " + node.item);
		}

		IntNode add(IntNode curr, Integer val){
			if(curr==null){
				curr = new IntNode(val);
				curr.size=1;
				return curr;
			}
			if(null==curr.left || null!=curr.right && curr.left.size.equals(curr.right.size)){
				curr.left = add(curr.left, val);
			}else {
				curr.right= add(curr.right, val);
			}
			curr.size=1;
			if(null!=curr.left) curr.size+=curr.left.size;
			if(null!=curr.right) curr.size+=curr.right.size;
			return curr;
		}
	}

	private static class IntNode {
		Integer item;
		IntNode right;
		IntNode left;
		Integer size;

		IntNode(Integer element){
			this.item=element;
		}

		IntNode(IntNode left, Integer element, IntNode right) {
			this.item = element;
			this.left = left;
			this.right = right;
		}
	}

	static void five(){
		//Copy linked list with arbitrary pointer
		Node<Integer> one = new Node<>(null, 1, null);
		Node<Integer> two = new Node<>(null, 10, one);
		Node<Integer> three = new Node<>(null, 11, two);
		Node<Integer> four = new Node<>(null, 13, three);
		Node<Integer> five = new Node<>(null, 7, four);
		one.prev=five;
		two.prev=three;
		three.prev=one;
		two.prev=five;

		Map<Node<Integer>, Node<Integer>> arbitraryMap = new HashMap<>();
		Node<Integer> temp = five;
		Node<Integer> newList = copyLL(temp, arbitraryMap);
		Node<Integer> newTemp = newList;
		while (temp!=null){
			newTemp.prev=arbitraryMap.get(temp);
			temp=temp.next;
			newTemp=newTemp.next;
		}

		System.out.println("Original:");
		while (five!=null){
			System.out.println(five.item);
			System.out.println(five);
			five=five.next;
		}
		System.out.println("Copy:");
		while (newList!=null){
			System.out.println(newList.item);
			System.out.println(newList);
			newList=newList.next;
		}
	}

	static Node<Integer> copyLL(Node<Integer> node, Map<Node<Integer>, Node<Integer>> arbitraryMap ){
		if (node==null)
			return null;
		Node<Integer> copy = new Node<>(null, node.item, null);
		copy.next = copyLL(node.next, arbitraryMap);
		arbitraryMap.put(node, copy);
		return copy;
	}

	static void testDLLCopy(){
		Node<Integer> one = new Node<>(null, 1, null);
		Node<Integer> two = new Node<>(null, 2, one);
		Node<Integer> three = new Node<>(null, 3, two);
		Node<Integer> four = new Node<>(null, 4, three);
		Node<Integer> five = new Node<>(null, 5, four);
		one.prev=two;
		two.prev=three;
		three.prev=four;
		four.prev=five;
		Node<Integer> copy = copyDLL(five, null);

		System.out.println("Original:");
		while (five!=null){
			System.out.println(five.item);
			if(five.prev!=null) System.out.println(five.prev.item);
			System.out.println(five);
			five=five.next;
		}
		System.out.println("Copy:");
		while (copy!=null){
			System.out.println(copy.item);
			if(copy.prev!=null) System.out.println(copy.prev.item);
			System.out.println(copy);
			copy=copy.next;
		}

	}

	static Node<Integer> copyDLL(Node<Integer> node, Node<Integer> prev){
		if (node==null)
			return null;
		Node<Integer> copy = new Node<>(prev, node.item, null);
		copy.next = copyDLL(node.next, copy);
		return copy;
	}

	static void four(){
		//Delete node with a given key
		Node<Integer> root = new Node<>(null, 9, null);
		root = new Node<>(null, 1, root);
		root = new Node<>(null, 5, root);
		root = new Node<>(null, 4, root);

		Integer key =5;
		Node<Integer> temp=root;
		if(root.item==key){
			root=root.next;
			while (root!=null){
				System.out.println(root.item);
				root=root.next;
			}
			return;
		}
		do {
			if (temp.next.item== key){
				temp.next=temp.next.next;
				while (root!=null){
					System.out.println(root.item);
					root=root.next;
				}
				return;
			}
			temp=temp.next;
		} while (temp.next!=null);
	}




	private static class Node<E> {
		E item;
		EmptyApplication.Node<E> next;
		EmptyApplication.Node<E> prev;

		Node(EmptyApplication.Node<E> prev, E element, EmptyApplication.Node<E> next) {
			this.item = element;
			this.next = next;
			this.prev = prev;
		}
	}

	static void three(){
		//Sum of two values
		int[] input = {2, 7, 11, 15};
		HashSet<Integer> aux = new HashSet<>();
		int target = 9;
		for (int i=0; i< input.length; i++){
			int diff = target-input[i];
			if (aux.contains(diff)){
				System.out.println("Found pair");
				return ;
			}else{
				aux.add(input[i]);
			}
		}
		System.out.println("Not found");
	}

	static void two(){
		//Find k closest numbers to x
		int k =3, x=6;
		int[] output = new int[k];

		Integer[] arr = {2, 4, 5, 6, 9};//{1, 2, 3, 4, 5};
		TreeSet<Integer> set = new TreeSet<>();
		Collections.addAll(set, arr);
		var smaller = set.headSet(x, true);
		var bigger = set.tailSet(x, false);
		for (int i =0; i<k; i++){
			if (x-smaller.last() <= bigger.first()-x){
				output[i]=smaller.pollLast();
				continue;
			}else{
				output[i]=bigger.pollFirst();
				continue;
			}
		}
		Arrays.sort(output);
		System.out.println(Arrays.toString(output));
	}

	static void one(){
		//Find the kth largest element in a number stream
		Queue<Integer> queue = new PriorityQueue<Integer>();
		add(queue,4);
		add(queue,1);
		add(queue,3);
		add(queue,12);
		add(queue,7);
		add(queue,14);

		add(queue, 6);
		System.out.println(queue.peek());
		add(queue, 13);
		System.out.println(queue.peek());
		add(queue, 4);
		System.out.println(queue.peek());
	}

	static void add(Queue<Integer> q, Integer num){
		q.add(num);
		if (q.size()>3) q.poll();
	}

	private static void test(){
		int[] array = new int[5];
		Integer[] integers = new Integer[5];

		Stack<Integer> stack = new Stack<>();

		Queue<Integer> queue = new PriorityQueue<>();
		queue = new LinkedList<>();

		List<Integer> list = new ArrayList<>();

		Set<Integer> set = new TreeSet<>();
		//set = new HashSet<>();
		//set = new LinkedHashSet<>();


		Map<Integer, Integer> map = new HashMap<>();
		map = new TreeMap<>();
		map = new LinkedHashMap<>();
		map = new IdentityHashMap<>();
		map = new Hashtable<>();

	}
}

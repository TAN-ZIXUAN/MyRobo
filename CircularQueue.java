package MyRobo;

import java.util.LinkedList;

public class CircularQueue<E> extends LinkedList {
    private int capacity = 10;

    public CircularQueue(int capacity){
        this.capacity = capacity;
    }

    public boolean add(Object e) {
        if(size() >= capacity)
            removeFirst();
        return super.add(e);
    }
}

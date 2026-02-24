public class pokerGame {
    public static void start() {
        // Create cards
        String[] poker = new String[54];
        String[] colors = {"♠", "♥", "♦", "♣"};
        String[] numbers = {"2", "3", "4", "5", "6", "7", "8", "9", "10", "J", "Q", "K", "A"};
        for (int i = 0; i < colors.length; i++) {
            for (int j = 0; j < numbers.length; j++) {
                poker[i * numbers.length + j] = colors[i] + numbers[j];
            }
        }
        poker[52] = "Big Joker";
        poker[53] = "Small Joker";
        System.out.println("Poker created:");
        for (int i = 0; i < poker.length; i++) {
            System.out.print(poker[i] + " ");
        }
        System.out.println();
        // Shuffle the cards
        for (int i = 0; i < poker.length; i++) {
            int index1 = (int) (Math.random() * poker.length);
            int index2 = (int) (Math.random() * poker.length);
            String temp = poker[index1];
            poker[index1] = poker[index2];
            poker[index2] = temp;
        }
        System.out.println("Poker shuffled:");
        for (int i = 0; i < poker.length; i++) {
            System.out.print(poker[i] + " ");
        }
        System.out.println();
    }
    public static void main(String[] args) {
        start();
    }
}
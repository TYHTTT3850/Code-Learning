public class test {
    public static void main(String[] args) {
        /*
        第一代手机：打电话
        第二代手机：打电话，发短信
        第三代手机：打电话升级为视频通话，发短信，玩游戏
        */

        FirstGenerationPhone phone1 = new FirstGenerationPhone();
        phone1.call();

        SecondGenerationPhone phone2 = new SecondGenerationPhone();
        phone2.call();
        phone2.sendMessage();

        ThirdGenerationPhone phone3 = new ThirdGenerationPhone();
        phone3.call();
        phone3.sendMessage();
        phone3.playGame();
    }
}
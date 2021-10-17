#include <Servo.h>

Servo myservo1;
Servo myservo2;

int angle1 = 140;
int angle2 = 90;
const int led = 13;
const int D0 = 2;
const int D1 = 3;
const int D2 = 4;
const int D3 = 5;

int a,b,c,d,data;


int Data()
{
  a = digitalRead(D0);
  b = digitalRead(D1);
  c = digitalRead(D2);
  d = digitalRead(D3);

  data = (8*d)+(4*c)+(2*b)+a;
  return data;
}

void setup() {
  
  myservo1.attach(8);
  myservo2.attach(9);
  
  Serial.begin(9600);

  pinMode(D0,INPUT_PULLUP);
  pinMode(D1,INPUT_PULLUP);
  pinMode(D2,INPUT_PULLUP);
  pinMode(D3,INPUT_PULLUP);
  pinMode(led,OUTPUT);

}



void loop() 
{     
      data = Data();

      if(data==1 && angle1<170){
        angle1 = angle1+2;
        myservo1.write(angle1);
     Serial.println("left");
      }
      else if(data==2 && angle1>60){
        angle1 = angle1 - 2;
        myservo1.write(angle1);
      Serial.println("right");
      }
      else if(data==3 && angle2<150){
        angle2 = angle2+2;
        myservo2.write(angle2);
     Serial.println("up");
      }
      else if(data==4 && angle2>90){
        angle2 = angle2 - 2;
        myservo2.write(angle2);
    Serial.println("down");
      }
      else if(data==5){
        digitalWrite(led,HIGH);
    Serial.println("led_on");
      }
      else if(data==6){
        digitalWrite(led,LOW);
    Serial.println("led_of");
      }
      else if(data==8){
        angle2 = 90;
        angle1 = 140;
        myservo2.write(angle2);
        myservo1.write(angle1);
    Serial.println("done");
      }
      delay(100);
 }

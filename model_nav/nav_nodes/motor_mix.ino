// Purpose: An improvised version detailing the process of moving at a certain degree of motion based on specified commands.
// lowest level
// Importing file Servo.h
#include <Servo.h>


// Set the pins to its byte value: an object of a given size and data.
byte servoPin_1 = 3;
byte servoPin_2 = 5;
byte servoPin_3 = 6;
byte servoPin_4 = 9;


// Define the servo objects.
Servo servo1;
Servo servo2;
Servo servo3;
Servo servo4;


// Sets a certain power value for moving forward, backward, and for stopping.
int motor_forward = 1550;
int motor_backward = 1450;
int motor_stop = 1500;


// Sets a certain character for degrees of motion when activating them.
const char move_forward = 'w';
const char move_backward = 'x';
const char stop_ = 's';
const char turn_left = 'a';
const char turn_right = 'd';


// Initialize Serial Communication.
void setup() {
  Serial.begin(115200);


// Servo to pin communication based on setting it to a specified pin.
  servo1.attach(servoPin_1);
  servo2.attach(servoPin_2);
  servo3.attach(servoPin_3);
  servo4.attach(servoPin_4);
  servo1.writeMicroseconds(1500);
  servo2.writeMicroseconds(1500);
  servo3.writeMicroseconds(1500);
  servo4.writeMicroseconds(1500);
  delay(1000);
}


// Print Move Forward.
Move forward based on commands.
void runCommand(char command) {
  if (command == turn_left) {
    servo1.write(motor_forward);
    servo2.write(motor_forward);
    servo3.write(motor_forward);
    servo4.write(motor_backward);


// Print Move Backward.
Move backward based on commands.
  } else if (command == turn_right) {
    //Serial.println("Move Backward");
    // Adjust the servo position to move back
    servo1.write(motor_backward);
    servo2.write(motor_backward);
    servo3.write(motor_backward);
    servo4.write(motor_forward);


// Print Turn Right.
Turn Right based on commands.
  } else if (command == move_forward) {
    //Serial.println("Turn Right");
    // Adjust the servo position to turn left
    servo1.write(1600);
    servo2.write(1600);
    servo3.write(1400);
    servo4.write(1600);


// Print Turn Left.
Turn Left based on commands.
  } else if (command == move_backward) {
    //Serial.println("Turn Left");
    // Adjust the servo position to turn right
    servo1.write(1400);
    servo2.write(1400);
    servo3.write(1600);
    servo4.write(1400);


// If none of the commands are chosen.
// Print Stop.
// Stop the boat based on commands.    
  } else if (command == stop_){
    Serial.println("Stop");
    // Adjust the servo position to stop
    servo1.write(motor_stop);
    servo2.write(motor_stop);
    servo3.write(motor_stop);
    servo4.write(motor_stop);
  }
}


// Serial commands are evaluated and then chosen based on a certain course of action by a certain command.
void loop() {
  // Check for serial commands and execute corresponding actions
  if (Serial.available()) {
    char command = Serial.read();
    runCommand(command);
  }
}


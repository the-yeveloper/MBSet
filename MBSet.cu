/* 
 * File:   MBSet.cu
 * 
 * Saket Gejji
 * 
 * Purpose:  This program displays Mandelbrot set using the GPU via CUDA and
 * OpenGL immediate mode.
 * 
 */

#include <iostream>
#include <stack>
#include <cuda_runtime_api.h>
#include <stdio.h>
#include "Complex.cu"
#include <algorithm>
#include <GL/freeglut.h>

// Size of window in pixels, both width and height
#define WINDOW_DIM            512

using namespace std;

// Initial screen coordinates, both host and device.
Complex minC(-2.0, -1.2);
Complex maxC(1.0, 1.8);
Complex* dev_minC;
Complex* dev_maxC;
Complex* dev_c;
const int maxIt = 2000; // Msximum Iterations
Complex* h_cArray;
int* h_pixelColours;
int* d_pixelColours;
Complex* d_cArray;
int x1, x2, yone, y2;
bool leftButton;
int startX, startY;
int endX, endY;
stack<double> cStack;
GLfloat updateRate = 20.0;

// Define the RGB Class
class RGB
{
public:
  RGB()
    : r(0), g(0), b(0) {}
  RGB(double r0, double g0, double b0)
    : r(r0), g(g0), b(b0) {}
public:
  double r;
  double g;
  double b;
};

RGB* colors = 0; // Array of color values
stack<int> mystack;

void InitializeColors()
{
  colors = new RGB[maxIt + 1];
  for (int i = 0; i < maxIt; ++i)
      colors[i] = RGB(i*0.024, i*0.016, i*0.008);
    
  colors[maxIt] = RGB(); // black
}

void drawPixel()
{
  for(int y=0; y<WINDOW_DIM; y++)
  {
    for(int x=0; x<WINDOW_DIM; x++)
    {
      glBegin(GL_POINTS);
       glColor3f(colors[h_pixelColours[y * WINDOW_DIM + x]].r, colors[h_pixelColours[y * WINDOW_DIM + x]].g, colors[h_pixelColours[y * WINDOW_DIM + x]].b);
       glVertex2f(x, y);
      glEnd();
    }
  }
}


__global__ void calculateIterations(int* d_pixelColours, Complex* dev_minC, Complex* dev_maxC)
{
  size_t index = blockIdx.x * blockDim.x + threadIdx.x;
  int x = index % WINDOW_DIM;
  int y = index / WINDOW_DIM;

  Complex dc(0, 0);
  dc.r = dev_maxC->r - dev_minC->r;
  dc.i = dev_maxC->i - dev_minC->i;

  float fx = (float)x / WINDOW_DIM;
  float fy = (float)y / WINDOW_DIM;
  
  
  Complex c(0,0);
  c.r = dev_minC->r +  (fx * dc.r);
  c.i = dev_minC->i +  (fy * dc.i);
  
  Complex z = c; 

  int it = 0;
  while(it < maxIt && z.magnitude2() < 2 * 2) 
  { 
    z = z * z + c;
    it++;
  }
  d_pixelColours[index] = it;
}

void callKernel()
{
  cudaMemcpy(dev_minC, &minC, sizeof(Complex), cudaMemcpyHostToDevice);
  cudaMemcpy(dev_maxC, &maxC, sizeof(Complex), cudaMemcpyHostToDevice);

  calculateIterations <<< (WINDOW_DIM * WINDOW_DIM /32), 32 >>>(d_pixelColours, dev_minC, dev_maxC);

  cudaMemcpy(h_pixelColours, d_pixelColours, WINDOW_DIM*WINDOW_DIM*sizeof(int), cudaMemcpyDeviceToHost);
}

void timer(int)
{
  glutPostRedisplay();
  glutTimerFunc(1000.0 / updateRate, timer, 0);
}

void drawSquare()
{
  glColor3f(1.0,0.0,0.0);
  glBegin(GL_LINE_LOOP);
     glVertex2f(startX, startY);
     glVertex2f(endX, startY);
     glVertex2f(endX, endY);
     glVertex2f(startX, endY);
  glEnd();
  glFlush();
  glutPostRedisplay();
  glutSwapBuffers();
}


void motion(int x, int y)
{

  endX = x;
  endY = 512-y;

  int dx = endX - startX;
  int dy = endY - startY;

  if(dx>0 && dy>0)
  {
    if(dx<dy)
      endY = startY+dx;
    else
      endX = startX+dy;
  }

  if(dx<0 && dy>0)
  {
    if((-1*dx)<dy)
      endY = startY+(-1*dx);
    else
      endX = startX+(-1*dy);
  }

  if(dx>0 && dy<0)
  {
    if(dx<(-1*dy))
      endY = startY+(-1*dx);
    else
      endX = startX+(-1*dy);
  }

  if(dx<0 && dy<0)
  {
    if(dx>dy)
      endY = startY+dx;
    else
      endX = startX+dy;
  }

  drawSquare();
  glutSwapBuffers();
}


void display(void)
{

  glClear(GL_COLOR_BUFFER_BIT);
  drawPixel();

  if(!leftButton)
  {
    drawSquare();
  }
  glutPostRedisplay();
  glutSwapBuffers();
}



void init()
{
  glutInitDisplayMode(GLUT_RGB | GLUT_SINGLE);
  glutInitWindowPosition(250, 250);
  glutInitWindowSize(WINDOW_DIM,WINDOW_DIM);
  glutCreateWindow("Mandelbrot Set");
  glClearColor(0.0, 0.0, 0.0, 0.0);
  glViewport(0, WINDOW_DIM, WINDOW_DIM, 0);
  glMatrixMode(GL_PROJECTION);
  glLoadIdentity();
  gluOrtho2D(0, WINDOW_DIM, 0, WINDOW_DIM);
  glMatrixMode(GL_MODELVIEW);
  glLoadIdentity();
}

void computeC()
{
  for(int y=0; y<WINDOW_DIM; y++)
  {
    for(int x=0; x<WINDOW_DIM; x++)
    {
      Complex dc(0, 0);
      dc.r = maxC.r - minC.r;
      dc.i = maxC.i - minC.i;
      float fx = (float)x / WINDOW_DIM;
      float fy = (float)y / WINDOW_DIM;
    
      Complex c = minC + Complex(fx * dc.r, fy * dc.i);

      h_cArray[y*WINDOW_DIM + x].r = c.r;
      h_cArray[y*WINDOW_DIM + x].i = c.i;
    }
  }
}

void mouse(int button, int state, int x, int y)
{

  leftButton=0;
  if((state == GLUT_DOWN) && (button == GLUT_LEFT_BUTTON))
  {
    x1 = x;
    yone = WINDOW_DIM-y;
    startX = x;
    startY = WINDOW_DIM-y;
    leftButton = 0;
    cStack.push(minC.r);
    cStack.push(minC.i);
    cStack.push(maxC.r);
    cStack.push(maxC.i);

    for(long int i=0; i<WINDOW_DIM*WINDOW_DIM; i++)
      mystack.push(h_pixelColours[i]);

    endX = x;
    endY = WINDOW_DIM-y;
  }

  if((state == GLUT_UP) && (button == GLUT_LEFT_BUTTON))
  {
    leftButton = 1;
  }

  if(leftButton==1)
  {
    int minX = min(startX, endX);
    int maxX = max(startX, endX);
    int minY = min(startY, endY);
    int maxY = max(startY, endY);

    minC.r = h_cArray[minY*WINDOW_DIM + minX].r;
    minC.i = h_cArray[minY*WINDOW_DIM + minX].i;
    maxC.r = h_cArray[maxY*WINDOW_DIM + maxX].r;
    maxC.i = h_cArray[maxY*WINDOW_DIM + maxX].i;

    computeC();
    callKernel();

    startX = 0;
    startY = 0;
    endX = 0;
    endY = 0;

    glutPostRedisplay();
  }
}



void keyboard (unsigned char key, int x, int y) 
{
  // Keystroke processing here
  if(key == 'b')
  {
    if(!cStack.empty() && !mystack.empty())
    {
      maxC.i = cStack.top();
      cStack.pop();
      maxC.r = cStack.top();
      cStack.pop();
      minC.i = cStack.top();
      cStack.pop();
      minC.r = cStack.top();
      cStack.pop();
  
      for(long int i=WINDOW_DIM*WINDOW_DIM; i>0; --i)
      {
        h_pixelColours[i] = mystack.top();
        mystack.pop();
      }
  
      computeC();
      callKernel();
      glutPostRedisplay();
    }
  }

  if(key=='q')
  {
    exit(0);
  }
}

int main(int argc, char** argv)
{
  glutInit(&argc, argv);
  init();

  h_pixelColours = (int*)malloc(WINDOW_DIM*WINDOW_DIM*sizeof(int));
  h_cArray = (Complex*)malloc(WINDOW_DIM*WINDOW_DIM*sizeof(Complex));

  cudaMalloc((void **)&d_pixelColours, WINDOW_DIM*WINDOW_DIM*sizeof(int));
  cudaMalloc((void **)&d_cArray, WINDOW_DIM*WINDOW_DIM*sizeof(Complex));

  cudaMalloc((void **)&dev_maxC, sizeof(Complex));
  cudaMalloc((void **)&dev_minC, sizeof(Complex));

  InitializeColors();
  computeC();
  callKernel();
  glutDisplayFunc(display);
  glutIdleFunc(display);
  glutKeyboardFunc (keyboard);
  glutMouseFunc(mouse);
  glutMotionFunc(motion);
  glutTimerFunc(1000.0 / updateRate, timer, 0);
  glutMainLoop(); // THis will callback the display, keyboard and mouse
  return 0;
  
}
